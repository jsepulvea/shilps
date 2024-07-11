import numpy as np
from enum import Enum

from power_grid_model import LoadGenType
from power_grid_model import PowerGridModel
from power_grid_model import initialize_array

class BUS_TYPE(Enum):
    SLACK   = 0
    PQ      = 1
    VOLTAGE = 2


class PFlow:

    def __init__(self, **kwargs):

        # Validate input
        # Check if bus_type, from_bus, to_bus, r_pu, x_pu are provided
        if ('bus_type' in kwargs and 'from_bus' in kwargs and
            'to_bus' in kwargs and 'r_pu' in kwargs and 'x_pu' in kwargs):

            # Assign values
            self.from_bus = kwargs["from_bus"]
            self.to_bus = kwargs["to_bus"]
            self.r_pu = kwargs["r_pu"]
            self.x_pu = kwargs["x_pu"]
            self.bus_type = kwargs["bus_type"]

        else:
            raise ValueError("Must provide either bus_type, from_bus, to_bus,"
                             " r_pu, and x_pu.")

        # Create backend model in power_grid_model
        self.backend_model = PFlow.line_info_to_backend(
            self.from_bus, self.to_bus, self.r_pu, self.x_pu, self.bus_type)

    @staticmethod
    def line_info_to_backend(from_bus, to_bus, r_pu, x_pu, bus_type):
                
        # line
        n_lines = len(from_bus)

        line = initialize_array('input', 'line', 1)
        line['id'] = np.asarray(range(n_lines))
        line['from_node'] = from_bus
        line['to_node'] = to_bus
        line['from_status'] = np.ones(n_lines, int)
        line['to_status'] = np.ones(n_lines, int)
        line['r1'] = r_pu
        line['x1'] = x_pu
        line['c1'] = 10e-6 * np.ones(n_lines)
        line['tan1'] = np.zeros(n_lines)
        line['i_n'] = 1e6 * np.ones(n_lines)

        # load
        sym_load = initialize_array('input', 'sym_load', 1)
        sym_load['id'] = [4]
        sym_load['node'] = [2]
        sym_load['status'] = [1]
        sym_load['type'] = [LoadGenType.const_power]
        sym_load['p_specified'] = [2e6]
        sym_load['q_specified'] = [0.5e6]
        
        # source
        source = initialize_array('input', 'source', 1)
        source['id'] = [5]
        source['node'] = [1]
        source['status'] = [1]
        source['u_ref'] = [1.0]
        # all
        self.backend_data = {
            'node': node,
            'line': line,
            'sym_load': sym_load,
            'source': source
        }

        self._first_run = False

        return backend_model

    def run(self, p_pu:np.ndarray, q_pu:np.ndarray):

        if self._first_run == False:
            


            self._first_run = True



    