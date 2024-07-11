from shilps.tiny_systemc import *
import shilps.legacy_pdata as legacy_pdata
from abc import ABC, abstractmethod

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
        
        self.legacy_build_module_from_pdata(pdata)

        self.register_task(EVENT.CLOCKTICK, Task(self.run_tick(), PRIORITY_PLANT))


    @staticmethod
    def read(case_folder: str, case_name: str):
        
        
        _t_start_time = time.time()
        pdata = legacy_pdata.Adn()
        pdata.read(case_folder, case_name)
        _t_end_time = time.time()

        _t_elapsed_time = round(_t_end_time - _t_start_time, 2)

        logging.info(f"Time spent in reading {case_name} case files: {_t_elapsed_time} seconds.")

        return DistributionNetwork(pdata)
    
    def legacy_build_module_from_pdata(self, pdata:legacy_pdata.Adn):
        """
        Implements a constructor for DistributionNetwork from lagacy pdata. 
        """
        # Create ports
        l_dgs = pdata.l_dgs
        l_buses0 = pdata.l_buses0

        for i in l_dgs:
            self.ports_pq_actuation[i] = Port(ChannelPQ, IOMODE.READ)

        for i in l_buses0:
            self.ports_bus_voltage[i] = Port(ChannelVoltageMeasurement, IOMODE.WRITE)


    def run_tick(self):
        # Solve a power flow
        self.compute_load_flow()
        # Update ports
        
        # Write to bus_voltage ports


        pass

    def compute_load_flow(self):
        

        pass

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
