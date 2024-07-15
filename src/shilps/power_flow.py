# TODO: Define a PFlow virtual class from which different backends could be
# implemented.

import numpy as np
from enum import Enum

import warnings

import pandas as pd

from power_grid_model import LoadGenType, ComponentType, DatasetType
from power_grid_model import PowerGridModel, CalculationMethod, CalculationType
from power_grid_model import initialize_array

from power_grid_model.validation import assert_valid_input_data, assert_valid_batch_data

class BUS_TYPE(Enum):
    SLACK   = 0
    PQ      = 1
    VOLTAGE = 2


class PowerFlow:

    def __init__(self, **kwargs):

        # Validate input
        # Check if bus_type, from_bus, to_bus, r_pu, x_pu are provided
        if ('bus_id' in kwargs and 'bus_type' in kwargs and 'from_bus' in kwargs and
            'to_bus' in kwargs and 'r_pu' in kwargs and 'x_pu' in kwargs):

            # Assign values
            self.from_bus = kwargs["from_bus"]
            self.to_bus = kwargs["to_bus"]
            self.r_pu = kwargs["r_pu"]
            self.x_pu = kwargs["x_pu"]
            self.bus_type = kwargs["bus_type"]
            self.bus_id = kwargs["bus_id"]

        else:
            raise ValueError("Must provide either bus_type, from_bus, to_bus,"
                             "r_pu, and x_pu.")

        self.buffer_input = None        
        self.buffer_output = None

        n_buses0 = len(self.bus_id)
        n_buses = n_buses0 - 1
        l_buses0 = range(n_buses0)

        self.buffer_update_pq = {
            ComponentType.sym_load: initialize_array(
                DatasetType.update, ComponentType.sym_load, n_buses)
        }

        
        # Index maps
        self.ar_load_indices = None

        # Create backend model in power_grid_model
        self.backend_model = self.line_info_to_backend(
            self.from_bus, self.to_bus, self.r_pu, self.x_pu, self.bus_type, self.bus_id)
        
    
    def line_info_to_backend(self, from_bus, to_bus, r_pu, x_pu, bus_type, bus_id):

        # Infer buses
        set_buses0 = list(set(from_bus).union(set(to_bus)))
        set_buses0.sort()

        assert all(x == i for i, x in enumerate(set_buses0, start=0))  # Sorted?

        n_buses0 = len(bus_type)
        n_buses = n_buses0 - 1

        node = initialize_array(DatasetType.input, ComponentType.node, n_buses0)
        node["id"] = bus_id
        node["u_rated"] = 12.5
          
        # line
        n_lines = len(from_bus)

        line = initialize_array(DatasetType.input, ComponentType.line, n_lines)
        line['id'] = np.asarray(range(n_buses0 + 1, n_buses0 +  n_lines + 1))
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
        n_comps = n_buses0 +  n_lines

        self.ar_load_indices = np.asarray(range(n_comps + 1, n_comps + n_buses + 1), int)

        sym_load = initialize_array(DatasetType.input, ComponentType.sym_load, n_buses)
        sym_load['id'] = self.ar_load_indices
        sym_load['node'] = np.asarray(range(1, n_buses + 1), int)
        sym_load['status'] = np.ones(n_buses, int)
        sym_load['type'] = [LoadGenType.const_power for _ in range(n_buses)]
        sym_load['p_specified'] = np.ones(n_buses)
        sym_load['q_specified'] = np.ones(n_buses)

        n_comps += n_buses
        
        # source
        source = initialize_array(DatasetType.input, ComponentType.source, 1)
        source['id'] = [n_comps + 1]
        source['node'] = [0]
        source['status'] = [1]
        source['u_ref'] = [1.0]

        # all
        input_data = {
            ComponentType.node: node,
            ComponentType.line: line,
            ComponentType.sym_load: sym_load,
            ComponentType.source: source
        }

        # assert_valid_input_data(input_data=input_data, calculation_type=CalculationType.power_flow)

        self.buffer_input = input_data
        return PowerGridModel(input_data)
    
    def update(self, busp_pu:np.ndarray, busq_pu:np.ndarray):
        
        self.buffer_update_pq[ComponentType.sym_load]["id"] = self.ar_load_indices
        self.buffer_update_pq[ComponentType.sym_load]["status"] = np.ones(busp_pu.shape[0] - 1, int)
        self.buffer_update_pq[ComponentType.sym_load]["p_specified"] = - busp_pu[1:]
        self.buffer_update_pq[ComponentType.sym_load]["q_specified"] = - busq_pu[1:]

        # assert_valid_batch_data(
        #     input_data=self.buffer_input,
        #     update_data=self.buffer_update_pq,
        #     calculation_type=CalculationType.power_flow
        # )

        self.backend_model.update(update_data=self.buffer_update_pq)

    def run(self):
        self.buffer_output = self.backend_model.calculate_power_flow(
            symmetric=True,
            error_tolerance=1e-8,
            max_iterations=100,
            calculation_method=CalculationMethod.newton_raphson)
        
    @property
    def v_pu(self):
        if self.buffer_output is not None:
            return self.buffer_output['node']["u_pu"]
        else:
            raise AttributeError("There is no v_pu to retrieve. Power flow has "
                                 "not been run yet!")
