"""
This file implements a version of the ieee1547 proportional droop controller
that is updated by a global control every hour in a dummy way. The objective is
to show how the different interfaces between modules work.

First, this file implements custom controller classes:
    1) 1 global controller
    2) 1 local controller for each PV generator

Second, a system is built by creating channels and binding ports.

Third, the simulation is executed.
"""
import numpy as np
import os
import random

import shilps.powersystems as ps

# **************************************************************************** #
# Load DistributionNetwork module
# **************************************************************************** #
THIS_DIR = os.path.dirname(__file__)
CASE_FOLDER = os.path.join(THIS_DIR, "data_seed/ieee4bus")
CASE_NAME = "ieee4bus"

config = {
    "time_ini": 0,
    "time_end": 10
}

shilps = ps.SimulationEnvironment(config)

grid = ps.DistributionNetwork.read(CASE_FOLDER, CASE_NAME)
shilps.add_module(grid)

# **************************************************************************** #
# The global update channel
# **************************************************************************** #
class ChannelIEEE1547GlobalLocal(ps.Channel):
    N_PARAMS = 8
    def __init__(self):
        self.ctrl_params = np.zeros((self.N_PARAMS,))
    
    def write(self, ctrl_params:np.ndarray):
        self.ctrl_params[:] = ctrl_params

    def read(self):
        return self.ctrl_params


# **************************************************************************** #
# The local controller
# Implements the global controller that updates the local controllers
# **************************************************************************** #
class CtrlAdaIEEE1547(ps.Controller):
    def __init__(self):
        super().__init__()
        # Controller parameters
        self.vl = 0.
        self.v1 = 0.
        self.v2 = 0.
        self.v3 = 0.
        self.v4 = 0.
        self.vh = 0.
        self.q1 = 0.
        self.q4 = 0.

        # Controller ports
        self.port_global_ctrl = ps.Port(ChannelIEEE1547GlobalLocal, ps.IOMODE.READ)
        self.port_voltage = ps.Port(ps.ChannelVoltageMeasurement, ps.IOMODE.READ)
        self.port_pq_actuation = ps.Port(ps.ChannelPQ, ps.IOMODE.WRITE)

        # Register tasks
        self.register_task(ps.EVENT.CLOCKTICK, ps.Task(self.run_tick, ps.PRIORITY_CTRL_LOCAL))

    def initialize(self):
        pass

    def actuate(self):
        # Read observations
        bus_voltage = self.port_voltage.read()

        # Implement droop control logic
        if bus_voltage < self.vl or bus_voltage > self.vh:
            reactive_power = 0  # No generation if out of bounds
        elif bus_voltage < self.v1:
            reactive_power = self.q1 * (bus_voltage - self.vl) / (self.v1 - self.vl)
        elif bus_voltage <= self.v4:
            reactive_power = self.q1 + ((self.q4 - self.q1) * (bus_voltage - self.v1) / (self.v4 - self.v1))
        else:
            reactive_power = self.q4

        # Actuate generator
        self.port_generator.write(reactive_power)


    def update(self):
        # Read parameters from global controller port
        (self.vl, self.v1, self.v2, self.v3, self.v4, self.vh, self.q1, self.q4
         ) = self.port_global_ctrl.read()


# **************************************************************************** #
# The global controller
# ---------------------
# Implements the global controller that updates the local controllers
# **************************************************************************** #
class GlobalController(ps.Controller):
    def __init__(self, l_dgs: list = None, grid: ps.DistributionNetwork = None):
        super().__init__()
        # Controller params
        if l_dgs is None:
            l_dgs = list(grid.ports_pq_actuation.keys())
        assert(l_dgs is not None)
        self.l_dgs = l_dgs
        self.update_interval = ps.timedelta(hours=1)

        self.register_task(ps.EVENT.CLOCKTICK, ps.Task(self.run_tick, ps.PRIORITY_CTRL_GLOBAL))

        # Controller ports
        self.ports_local_controllers = {
            idx_dg: ps.Port(ChannelIEEE1547GlobalLocal, ps.IOMODE.WRITE) for idx_dg in l_dgs}

    def initialize(self):
        self.update_parameters()
        self.schedule_next_update()

    def schedule_next_update(self):
        self.add_event(ps.SimulationTimeDeltaEvent(self.update_interval, self.update_parameters))

    def actuate(self):
        """
        Updates the control parameters with dummy values and writes them to the interface.
        """
        ps.logging.debug("GlobalController actuate called." )
        # Dummy parameter update logic: random values within a range
        new_params = np.array([
            random.uniform(0.90, 0.95),  # vl
            random.uniform(0.95, 1.00),  # v1
            random.uniform(1.00, 1.05),  # v2
            random.uniform(1.05, 1.10),  # v3
            random.uniform(1.10, 1.15),  # v4
            random.uniform(1.15, 1.20),  # vh
            random.uniform(-0.5, 0.0),   # q1
            random.uniform(0.0, 0.5)     # q4
        ])

        for port in self.ports_local_controllers.values():
            port.write(new_params)

    def update(self):
        print("Update Global Controller")



# **************************************************************************** #
# Build the system by linking the modules (create links and bind ports)
# **************************************************************************** #

global_controller = GlobalController(grid=grid)
shilps.add_module(global_controller)
for idx_dg, dg_actuation_port in grid.ports_pq_actuation.items():
    
    # Create local controller
    local_controller = CtrlAdaIEEE1547()
    shilps.add_module(local_controller)

    # Bind local controller
    channel_local = ps.ChannelPQ()
    local_controller.port_pq_actuation.bind(channel_local)
    dg_actuation_port.bind(channel_local)

    # Bind global controller 
    channel_global = ChannelIEEE1547GlobalLocal()
    global_controller.ports_local_controllers[idx_dg].bind(channel_global)
    local_controller.port_global_ctrl.bind(channel_global)

    # Bind voltage measurement devices
    #grid.pdata.dgs[idx_dg, ]
    bus_id = grid.pdata.dgs.loc[idx_dg, "bus"]
    channel_vmeasure = ps.ChannelVoltageMeasurement()
    local_controller.port_voltage.bind(channel_vmeasure)
    grid.ports_bus_voltage[bus_id].bind(channel_vmeasure)


# **************************************************************************** #
# Config and run simulation
# **************************************************************************** #
# Set up the clock
shilps.run()
