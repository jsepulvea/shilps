from shilps.tiny_systemc import *
import shilps.legacy_pdata as legacy_pdata
from abc import ABC, abstractmethod
import shilps.power_flow as pf

import numpy as np
from scipy.sparse import csr_matrix

from datetime import timedelta
import time

PRIORITY_CTRL_GLOBAL = 0
PRIORITY_CTRL_LOCAL = 1
PRIORITY_PLANT = 2

# **************************************************************************** #
# Links
# **************************************************************************** 
class ChannelVoltageMeasurement(Channel):
    def __init__(self) -> None:
        super().__init__()
        self.v: float = None

    def read(self):
        return self.v

    def write(self, v: float):
        self.v = v

class ChannelPQ(Channel):
    def __init__(self) -> None:
        super().__init__()
        self.p: float = None
        self.q: float = None

    def read(self):
        return (self.p, self.q)

    def write(self, p: float, q: float):
        self.p = p
        self.q = q

# **************************************************************************** #
# Modules
# **************************************************************************** #
class DistributionNetwork(Module):
    """
    TODO: Refactor to avoid utilizing legacy code!
    """
    
    def __init__(self, pdata: legacy_pdata.Adn) -> None:
        super().__init__()

        self.pdata = pdata

        # Port containers
        self.ports_pq_actuation = {}
        self.ports_bus_voltage = {}

        for i in self.l_dgs:
            self.ports_pq_actuation[i] = Port(ChannelPQ, IOMODE.READ)

        for i in self.l_buses0:
            self.ports_bus_voltage[i] = Port(ChannelVoltageMeasurement, IOMODE.WRITE)

        self.register_task(EVENT.CLOCKTICK, Task(self.run_tick, PRIORITY_PLANT))

        self.power_flow = self.build_power_flow_model()

        
        n_dgs = len(self.l_dgs)
        n_buses0 = len(self.l_buses0)

        self._l_cols_loadp = [colname for colname in self.pdata.df_data.columns if colname.startswith('loadp')]
        self._l_cols_loadq = [colname for colname in self.pdata.df_data.columns if colname.startswith('loadq')]
        self._l_cols_dgs = ['dgpmax{}'.format(i) for i in self.l_dgs]
        n_loads = len(self._l_cols_loadp)

        # Memory allocation buffers for repeated operations
        self._dgs_pmax = np.zeros(n_dgs)
        self._dgs_qmax = np.zeros(n_dgs)

        self._dgp_ctrl = np.zeros(n_dgs)
        self._dgq_ctrl = np.zeros(n_dgs)

        self._dgs_pproj = np.zeros(n_dgs) 
        self._dgs_qproj = np.zeros(n_dgs)

        self._bus0_dgs_pproj = np.zeros(n_dgs)
        self._bus0_dgs_qproj = np.zeros(n_dgs)

        self._pbus0 = np.zeros(shape=(n_buses0, 1))
        self._qbus0 = np.zeros(shape=(n_buses0, 1))
        self._loadp = np.zeros(n_loads)
        self._loadq = np.zeros(n_loads)
        self._busloadp = np.zeros(n_loads)
        self._busloadq = np.zeros(n_loads)
        
        

        self.dgs_snom = self.get_dgs_snom()

        # Incidence matrices
        self.incidence_dgs2buses0 = None
        self.incidence_loads2buses0 = None
        self.make_incidence_matrices()

    def make_incidence_matrices(self):
        # Construct a dg to bus incidence matrix
        n_dgs = len(self.l_dgs)
        n_buses0 = len(self.l_buses0)
        
        row_indices = self.pdata.dgs.loc[:, "bus"].values
        col_indices = list(range(len(self.l_dgs)))
        data = np.ones(n_dgs)
        self.incidence_dgs2buses0 = csr_matrix(
            (data, (row_indices, col_indices)), shape=(n_buses0, n_dgs))
        
        # Construct load2buses incidence matrix
        l_cols_loadp = self._l_cols_loadp
        n_loads = len(l_cols_loadp)
        row_indices = [int(colname.split('loadp')[1]) for colname in l_cols_loadp]
        col_indices = list(range(len(l_cols_loadp)))
        data = np.ones(n_loads)
        self.incidence_loads2buses0 = csr_matrix(
            (data, (row_indices, col_indices)), shape=(n_buses0, n_dgs))

    @staticmethod
    def read(case_folder: str, case_name: str):
        
        
        _t_start_time = time.time()
        pdata = legacy_pdata.Adn()
        pdata.read(case_folder, case_name)
        _t_end_time = time.time()

        _t_elapsed_time = round(_t_end_time - _t_start_time, 2)

        logging.info(f"Time spent in reading {case_name} case files: {_t_elapsed_time} seconds.")

        return DistributionNetwork(pdata)
    
    def run_tick(self):
        # Read from controller references
        # TODO: Indices mapping!!!!
        for i, idx in enumerate(self.l_dgs):
            self._dgp_ctrl[i], self._dgq_ctrl[i] = self.ports_pq_actuation[idx].read()
            
        # Solve a power flow
        self.compute_power_flow()
        # Update ports with the power_flow output
        # TODO: Here there should be a mapping process from indices in the
        # backend to indices in the frontend.
        v_pu = self.power_flow.v_pu

        for (id, v) in enumerate(v_pu):
            self.ports_bus_voltage[i].write(v)


    def build_power_flow_model(self):
        """
        Constructs a :obj:`shilps.power_flow.PowerFlow` model from
        `shilps.powersystems.DistributionNetwork`.
        """
        bus_type = self.bus_type
        from_bus = self.from_bus
        to_bus = self.to_bus
        r_pu = self.r_pu
        x_pu = self.x_pu
        bus_id = self.bus_id

        return pf.PowerFlow(bus_id = bus_id, bus_type=bus_type,
                            from_bus=from_bus, to_bus=to_bus, r_pu=r_pu, x_pu=x_pu)

    def compute_power_flow(self):

        logging.debug("DistributionNetwork.compute_power_flow called.")

        # Update power flow
        current_time = self.get_time()

        self._dgs_pmax[:] = self.get_dgpmax(current_time)
        np.minimum(self._dgp_ctrl, self._dgs_pmax, out=self._dgs_pproj)
        np.sqrt(np.power(self.dgs_snom, 2) - np.power(self._dgs_pproj, 2), out=self._dgs_qmax)
        np.minimum(self._dgq_ctrl, self._dgs_qmax, out=self._dgs_qproj)

        self._bus0_dgs_pproj = self.incidence_dgs2buses0.dot(self._dgs_pproj.reshape(-1, 1))
        self._bus0_dgs_qproj = self.incidence_dgs2buses0.dot(self._dgs_qproj.reshape(-1, 1))

        self._busloadp = self.incidence_loads2buses0.dot(self.get_loadp(current_time))
        self._busloadq = self.incidence_loads2buses0.dot(self.get_loadq(current_time))

        np.add(self._bus0_dgs_pproj, - self._busloadp, out=self._pbus0)
        np.add(self._bus0_dgs_qproj, - self._busloadq, out=self._qbus0)

        self.power_flow.update(self._pbus0.reshape(-1), self._qbus0.reshape(-1))

        # Run power flow
        self.power_flow.run()
        

    def get_dgs_ports(self):
        """
        TODO: This should be implemented as an iterator
        """
        pass

    def validate(self):
        """
        Raises an error if not valid.
        """
        # Check if the number of dg related ports coincide in number
        pass

    
    @property
    def bus_type(self):
        """
        TODO: This should come from the data structure
        Returns a numpy array of integers
        """
        n_buses0 = self.n_buses0
        return np.concatenate([np.asarray([pf.BUS_TYPE.SLACK]),
                                   np.asarray([pf.BUS_TYPE.PQ for _ in range(n_buses0 - 1)])])

    @property
    def l_dgs(self):
        return self.pdata.l_dgs
    
    def get_dgs_snom(self):
        return self.pdata.dgs.loc[:, "snom"].to_numpy()
    
    @property
    def l_buses0(self):
        return self.pdata.l_buses0
    
    @property
    def n_buses0(self):
        return len(self.pdata.l_buses0)

    @property
    def from_bus(self):
        return self.pdata.branches.loc[:, "busf"].to_numpy()
    
    @property
    def to_bus(self):
        return self.pdata.branches.loc[:, "bust"].to_numpy()
    
    @property
    def r_pu(self):
        return self.pdata.branches.loc[:, "r"].to_numpy()
    
    @property
    def x_pu(self):
        return self.pdata.branches.loc[:, "x"].to_numpy()
    
    @property
    def bus_id(self):
        return self.pdata.buses.index.to_numpy()
    
    def get_loadp(self, idx_time):
        """ Returns the vector of active loads ordered according to the load
        indices
        """
        return self.pdata.df_data.loc[idx_time, self._l_cols_loadp].values.reshape(-1, 1)
    
    def get_loadq(self, idx_time):
        """ Returns the vector of reactive loads ordered according to the load
        indices.
        """
        return self.pdata.df_data.loc[idx_time, self._l_cols_loadq].values.reshape(-1, 1)
    
    def get_dgpmax(self, idx_time):
        """Returns the vector of dgs pmax ordered according to the dgs indices.
        """

        return self.pdata.df_data.loc[idx_time, self._l_cols_dgs].values


class Generator(Module):
    def __init__(self, snom_MVA) -> None:
        self.snom_MVA = snom_MVA


class Controller(Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def update(self):
        pass
    
    @abstractmethod
    def actuate(self, observations):
        pass

    def run_tick(self):
        self.update()
        self.actuate()
