import numpy as np
from enum import Enum
from abc import ABC, abstractmethod
from typing import Callable

from .global_definitions import SUCCESS_STRING

from .shilps_logging import logging

import queue

class IOMODE(Enum):
    WRITE = 1
    READ = 2

class EVENT(Enum):
    INITIALIZE = 1
    CLOCKTICK = 2
    FINISH = 3

MAX_AMOUNT_TASKS = 1000


class Task:
    """
    TODO: Remove the priority_hint field
    TODO: write more meaningful taks printing.
    """
    def __init__(self, task: Callable[[], None], priority_hint: int = 0):
        self.proc = task
        self.priority_hint = priority_hint

    def run(self):
        self.proc()

    def __str__(self):
        return f"Task(priority_hint={self.priority_hint})"

    def __repr__(self):
        return f"Task(task={self.proc.__name__}, priority_hint={self.priority_hint})"


class Channel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def write(self, data):
        pass    

class Port:
    def __init__(self, channel_type: type, mode: IOMODE):
        self.mode = mode
        self.channel_type = channel_type
        self.channel = None

    def read(self):
        return self.channel.read()
    
    def write(self, *args, **kwargs) -> None:
        self.channel.write(*args, **kwargs)

    def bind(self, channel: Channel):
        if not isinstance(channel, self.channel_type):
            raise TypeError("Port is binding to a link of different type")
    
        self.channel = channel

    def isbinded(self):
        return self.channel is not None

class InvalidModule(Exception):
    """Exception raised when a InvalidModule instance is invalid."""
    def __init__(self, message, module):
        super().__init__(f"Invalid {type(module)} instance : {message}")

class Module:
    def __init__(self):
        self.get_time = None
        self._port_idx_counter = 0
        self.ports = {}
        self.event2task = {i: [] for i in EVENT}

    def add_port(self, port:Port):
        self._port_idx_counter += 1
        self.ports[self._port_idx_counter] = port
        return self._port_idx_counter
    
    def register_task(self, event:EVENT, task:Task):
        self.event2task[event].append(task)

        

class SimulationKernel(ABC):
    
    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def add_module(self, module: Module):
        pass

class UnknownEventError(Exception):
    """Exception raised when an unknown event is encountered."""
    def __init__(self, event):
        super().__init__(f"Unknown event encountered: {event}")


class PrioritySequentialKernel(SimulationKernel):
    """
    Kernel and scheduler are the same?

    Takes care about:
        - Management
        - Synchronisation
        - Load balancing (This no)
    """
    NAME = "Priority-based sequential kernel"

    def __init__(self, config: dict=None):
        
        super().__init__()

        # Time config
        self.time_ini = None 
        self.time_end = None
        self.time_delta = None
        self.current_time = None

        if "time_config" in config:
            self.set_time_config(config["time_config"])

        self.modules = []

        self.event2task_priority = {event: [] for event in EVENT}

        self.is_updated = False

        # Create a FIFO queue
        self.event_queue = queue.Queue()

        self._started = False
        
    def set_time_config(self, time_config):
        self.time_ini = time_config["time_ini"]
        self.time_end = time_config["time_end"]
        self.time_delta = time_config["time_delta"]
        self.current_time = time_config["time_ini"]
        if self._started == False:
            logging.info(f"{PrioritySequentialKernel.NAME} is ready to start.")
            self._started = True
        else:
            logging.info(f"{PrioritySequentialKernel.NAME} time_config reset.")

    def get_time(self):
        return self.current_time

    def add_module(self, module: Module):
        assert(isinstance(module, Module))
        assert(module is not None)
        
        module.get_time = self.get_time
        self.modules.append(module)
        if self.is_updated:
            self.is_updated = False
    
    def put_event(self, event:EVENT):
        self.event_queue.put(event)

    def get_event(self):
        """
        TODO: Change the get_nowait for asynchronic implementation.
        """
        if self.event_queue.qsize() > 0:
            return self.event_queue.get()
        else:
            return None
    
    def handle_clocktick_event(self):
        # TODO: Avoid the use of priority hints and build a better kernel.
        # Collect all the CLOCKTICK tasks and sort them by the give priority hint

        for task in self.event2task_priority[EVENT.CLOCKTICK]:
            task.run()

        
        self.current_time += self.time_delta
        if self.current_time <= self.time_end:
            self.event_queue.put(EVENT.CLOCKTICK)
        else:
            self.current_time -= self.time_delta
            self.event_queue.put(EVENT.FINISH)
        

    def handle_initialize_event(self):
        
        for task in self.event2task_priority[EVENT.INITIALIZE]:
            task.run()
        
        self.event_queue.put(EVENT.CLOCKTICK)

    def handle_finish_event(self):
        
        for task in self.event2task_priority[EVENT.FINISH]:
            task.run()

    def run(self):
        assert self.is_updated

        logging.info("HardCodedSequentialKernel run method called.")

        self.put_event(EVENT.INITIALIZE)

        while True:
            logging.info(f"The kernel event_queue size is {self.event_queue.qsize()}.")
            
            event = self.get_event()
            time_format = "%y-%m-%d %H:%M:%S"
            logging.info(f"Handling event {event}. Time: {self.current_time.strftime(time_format)}")

            if event == EVENT.INITIALIZE:
                self.handle_initialize_event()
            elif event == EVENT.CLOCKTICK:
                self.handle_clocktick_event()
            elif event == EVENT.FINISH: 
                self.handle_finish_event()
            elif event == None:
                logging.info(f"Simulation finished sucessfully{SUCCESS_STRING}.")
                break
            else:
                raise UnknownEventError(event)
            
    def update(self):
        """
        Creates the global event to list of tasks object from modules
        """
        
        # Iterate through modules and add the registered tasks to the
        # self.event2task_priority field.

        for m in self.modules:
            for (e, list_of_tasks) in m.event2task.items():
                for t in list_of_tasks:
                    self.event2task_priority[e].append(t)

        # Sort
        for k in self.event2task_priority.keys():
            self.event2task_priority[k].sort(key=lambda task: task.priority_hint)
        
        self.is_updated = True

class SimulationEnvironment:
    """
    A class to represent a simulation environment where modules can be
    dynamically added.

    TODO: Change the module indexing methodology
    """

    def __init__(self, config: dict={}, kernel: SimulationKernel = None):
        """
        Constructs all the necessary attributes for the simulation environment
        object.
        """

        self.config = config
        
        if kernel is not None:
            self.kernel = kernel
        else:
            self.kernel = self._default_simulation_kernel(self.config)
            
        self.modules = {}
        self._idx_counter_modules = 0

    def _new_index(self):
        """
        Increment the internal index counter and return the new index.

        Returns
        -------
        int
            The next index number that will be assigned to a new module.
        """
        self._idx_counter_modules += 1
        return self._idx_counter_modules

    def add_module(self, module):
        """
        Add a module to the environment and assign it a unique index.

        Parameters
        ----------
        module : any
            The module to add to the environment. The type of this parameter depends on the specifics of the simulation.

        Returns
        -------
        int
            The index number assigned to the newly added module.
        """
        k = self._new_index()
        self.modules[k] = module
        # Pass it to the simulation kernel
        self.kernel.add_module(module)

        return k
    
    def _default_simulation_kernel(self, config):
        """
        Constructs default simulation kernel.
        """
        return PrioritySequentialKernel(config)
    
    def set_time_config(self, time_config: dict):
        self.config["time_config"] = time_config
        self.kernel.set_time_config(time_config)
 
    def run(self, time_config: dict = None):
        if time_config is not None:
            self.set_time_config(time_config)

        self.kernel.update()
        logging.info("Simulation kernel updated.")
        self.kernel.run()

    @property
    def time_config(self):
        return self.config["time_config"]
