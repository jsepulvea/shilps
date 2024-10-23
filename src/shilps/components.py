from dataclasses import dataclass, field
import pandas as pd
from typing import Dict, Any, Type, List, Tuple
import json
import numpy as np
import os
import glob

ISO8601 = "%Y-%m-%dT%H:%M:%S%z"


@dataclass
class SerializableDataClass:
    def to_dict(self) -> Dict[str, Any]:
        """Convert the component's data to a dictionary for easy DataFrame
        insertion.

        If a field has a to_dict method, it will be called.
        """
        data = {}
        for field in self.__dataclass_fields__.values():
            value = self.__getattribute__(field.name)
            if hasattr(value, "to_dict"):
                data[field.name] = value.to_dict()
            else:
                data[field.name] = value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create a component from a dictionary.

        If a field has a from_dict method, it will be called.
        """
        fields = cls.__dataclass_fields__.values()
        kwargs = {}
        for field in fields:
            value = data[field.name]
            if hasattr(field.type, "from_dict"):
                kwargs[field.name] = field.type.from_dict(value)
            else:
                kwargs[field.name] = value

        return cls(**kwargs)

    def copy(self):
        kwargs = {}
        for field in self.__dataclass_fields__.values():
            value = self.__getattribute__(field.name)
            if hasattr(value, "copy"):
                value = value.copy()
            kwargs[field.name] = value

        return self.__class__(**kwargs)


@dataclass(slots=True)
class Component(SerializableDataClass):
    index: int = None
    name: str = None

class TimeConfig:
    def __init__(self, start=None, end=None, periods=None, freq=None, tz=None):
        self.start = start
        self.end = end
        self.periods = periods
        self.freq = freq
        self.tz = tz

    def _create_range(self):
        return pd.date_range(
            start=self.start,
            end=self.end,
            periods=self.periods,
            freq=self.freq,
            tz=self.tz,
        )

    @property
    def range(self):
        return self._create_range()

    def to_dict(self):
        # Convert datetime objects to ISO format strings
        return {
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "periods": self.periods,
            "freq": self.freq,
            "tz": self.tz,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        # Convert ISO format strings back to datetime objects
        data["start"] = (
            pd.to_datetime(data["start"]) if data.get("start") else None
        )
        data["end"] = pd.to_datetime(data["end"]) if data.get("end") else None
        return cls(**data)


@dataclass(slots=True)
class PlanningTimeConfig(SerializableDataClass):
    """
    Configuration class for defining time-related parameters in a planning
    horizon.

    Parameters
    ----------
    planning_horizon_start : str or pandas.Timestamp
        The start date of the planning horizon. If a string is provided, it will
        be converted to a `pandas.Timestamp`.
    planning_horizon_end : str or pandas.Timestamp
        The end date of the planning horizon. If a string is provided, it will
        be converted to a `pandas.Timestamp`.
    scenarios_per_year : int
        Number of representative scenarios to generate per year.
    scenario_resolution : str
        Temporal resolution of each scenario. For example, '1D' represents daily
        scenarios, '1W' represents weekly scenarios, etc.
    scenario_subsampling : str
        Subsampling frequency used for dynamic modeling purposes, such as
        battery simulation within the defined scenario. For example, '1h' for
        hourly subsampling.

    Raises
    ------
    ValueError
        If `planning_horizon_end` is earlier than or equal to
        `planning_horizon_start`.

    Examples
    --------
    Create a time configuration for a planning horizon from 2025 to 2065 with
    daily scenarios and hourly subsampling:

    ```python
    time_config = TimeConfig(
        planning_horizon_start='2025-01-01',
        planning_horizon_end='2065-12-31',
        scenarios_per_year=12,
        scenario_resolution='1D',
        scenario_subsampling='1h'
    )
    ```
    """
    planning_horizon_start: pd.Timestamp  
    planning_horizon_end: pd.Timestamp    
    scenarios_per_year: int               
    scenario_resolution: str              
    scenario_subsampling: str             

    def __post_init__(self):
        # Ensure that planning_horizon_start and planning_horizon_end are valid
        # pandas Timestamps
        if isinstance(self.planning_horizon_start, str):
            self.planning_horizon_start = pd.Timestamp(self.planning_horizon_start)
        if isinstance(self.planning_horizon_end, str):
            self.planning_horizon_end = pd.Timestamp(self.planning_horizon_end)

        # Validate that the planning horizon end is after the start
        if self.planning_horizon_end <= self.planning_horizon_start:
            raise ValueError("planning_horizon_end must be after planning_horizon_start.")

    @property
    def range(self) -> pd.DatetimeIndex:
        """
        Returns the range of dates from planning_horizon_start to
        planning_horizon_end at the resolution specified by scenario_resolution.

        Returns
        -------
        pd.DatetimeIndex
            A range of dates between `planning_horizon_start` and
            `planning_horizon_end` using the frequency defined in
            `scenario_resolution`.

        Examples
        --------
        >>> time_config = TimeConfig(
        ...     planning_horizon_start='2025-01-01',
        ...     planning_horizon_end='2025-12-31',
        ...     scenarios_per_year=12,
        ...     scenario_resolution='1D',
        ...     scenario_subsampling='1h'
        ... )
        >>> time_config.range
        DatetimeIndex(['2025-01-01', '2025-01-02', ..., '2025-12-31'], dtype='datetime64[ns]', freq='D')
        """
        return pd.date_range(start=self.planning_horizon_start, 
                             end=self.planning_horizon_end, 
                             freq=self.scenario_resolution)

    @property
    def scenario_duration(self) -> pd.Timedelta:
        return pd.Timedelta(self.scenario_resolution)
    
    @property
    def planning_horizon_duration(self) -> pd.Timedelta:
        return self.planning_horizon_end - self.planning_horizon_start

def generate_scenario_time_ranges(time_config: PlanningTimeConfig) -> Dict[Tuple[int, int], pd.DatetimeIndex]:
    """
    Generate a set of time ranges indexed by a tuple (year, scenario), where
    each time range corresponds to a randomly selected time period within
    the year, using the scenario resolution and scenario subsampling
    frequency from the TimeConfig.

    Parameters
    ----------
    time_config : TimeConfig
        The configuration that defines the planning horizon, scenario
        resolution, and subsampling frequency.

    Returns
    -------
    Dict[Tuple[int, int], pd.DatetimeIndex]
        A dictionary where the keys are tuples (year, scenario), and the
        values are the corresponding time ranges with subsampling frequency.
    """
    time_ranges = {}
    # Raise error if the planning horizon does not end in 12-31
    if (time_config.planning_horizon_end.month != 12
        or time_config.planning_horizon_end.day != 31):
        raise ValueError("The planning horizon must end on December 31st.")

    # Get the list of years in the planning horizon
    years = pd.date_range(time_config.planning_horizon_start, 
                          time_config.planning_horizon_end, 
                          freq='YE').year

    # Loop over each year
    for year in years:
        # Generate scenarios for this year
        for scenario in range(time_config.scenarios_per_year):
            # Define the start and end dates for this year
            year_start = pd.Timestamp(f"{year}-01-01")
            year_end = pd.Timestamp(f"{year}-12-31")

            # Randomly select a start date within the year for the scenario
            random_start = pd.Timestamp(np.random.choice(pd.date_range(year_start, year_end, freq='D')))
            
            # Generate a time range with the scenario_resolution and scenario_subsampling
            time_range = pd.date_range(start=random_start, 
                                       end=random_start + pd.Timedelta(time_config.scenario_resolution),
                                       inclusive="left",
                                       freq=time_config.scenario_subsampling)

            # Store the time range in the dictionary with (year, scenario) as the key
            time_ranges[(year, scenario)] = time_range

    return time_ranges

class DataScenarioTime:
    def __init__(self, df: pd.DataFrame = None, time_config: TimeConfig = None):

        self.df = df
        self.time_config = time_config

    def get_value(self, tsname, scenario, time):
        return self.df.loc[(scenario, time), tsname]

    def set_value(self, tsname, scenario, time, value):
        self.df.loc[(scenario, time), tsname] = value

    def __contains__(self, tsname):
        return tsname in self.df.columns

    def write(self, json_path: str, csv_path: str):
        """Write the DataFrame to a CSV file and the time configuration to a
        JSON file.

        :param json_path: Path to save the JSON file containing time
            configuration.
        :param csv_path: Path to save the CSV file containing the DataFrame.
        """
        # Ensure the MultiIndex levels are named 'scenario' and 'time'
        self.df.index.names = ["scenario", "time"]

        # Convert the 'time' index to the required format
        self.df.index = self.df.index.set_levels(
            [
                self.df.index.levels[0],  # scenario remains the same
                self.df.index.levels[1].strftime(
                    "%Y-%m-%d %H:%M:%S %Z"
                ),  # format time as y-m-d HH:MM:SS tz
            ]
        )

        # Save the time_config as JSON
        with open(json_path, "w") as json_file:
            json.dump(self.time_config.to_dict(), json_file)

        # Save the DataFrame as CSV
        self.df.to_csv(csv_path)

    def read(self, json_path: str, csv_path: str):
        """Read the DataFrame from a CSV file and the time configuration from a
        JSON file.

        :param json_path: Path to the JSON file containing time configuration.
        :param csv_path: Path to the CSV file containing the DataFrame.
        """
        # Load the time_config from JSON
        with open(json_path, "r") as json_file:
            self.time_config = TimeConfig.from_dict(json.load(json_file))

        # Load the DataFrame from CSV
        self.df = pd.read_csv(csv_path, index_col=[0, 1])

        # Ensure the MultiIndex levels are named 'scenario' and 'time'
        self.df.index.names = ["scenario", "time"]

        # Parse the 'time' index back to a datetime object with the correct
        # timezone
        self.df.index = self.df.index.set_levels(
            [
                self.df.index.levels[0],  # scenario remains the same
                pd.to_datetime(
                    self.df.index.levels[1], format="%Y-%m-%d %H:%M:%S %Z"
                ),  # parse time back
            ]
        )


class DataTimeSeries:
    def __init__(self, dict_df: Dict[Any, pd.DataFrame] = None, tsnames: str = None,
                 time_config:TimeConfig = None, scenarios: list = None,
                 time_ranges: dict = None):
        
        self.tsnames = tsnames
        self.dict_df = dict_df if dict_df is not None else {}

        if time_config is not None and tsnames is not None and scenarios is not None:
            for scenario in scenarios:
                self.dict_df[scenario] = self._create_empty_df(tsnames, time_config)
        
        if time_ranges is not None:
            for key, time_range in time_ranges.items():
                self.dict_df[key] = self._create_empty_df(tsnames, time_range)
        
    @staticmethod
    def _create_empty_df(tsnames, time_range):
        return pd.DataFrame(index=time_range, columns=tsnames)
    
    def get_value(self, tsname, *args):
        scenario = args[0]
        key = args[1:]
        return self.dict_df[scenario].loc[key, tsname]

    def get_series(self, *args):
        """Retrieve a specific time series from the DataFrame corresponding to a
        key.

        Parameters
        ----------
        *args : tuple
            A variable-length argument list where all elements except the last
            one are used to form the key (as a tuple) for accessing the
            dictionary. The last element of `args` is the name of the time
            series to retrieve from the DataFrame.

        Returns
        -------
        pandas.Series
            The requested time series from the DataFrame associated with the
            specified key.

        Raises
        ------
        KeyError
            If the specified key does not exist in `self.dict_df`.
        """
        dict_key = tuple(args[:-1])
        tsname = args[-1]
        return self.dict_df[dict_key].loc[:, tsname]

    def set_series(self, *args, value):
        """Set a specific time series in the DataFrame corresponding to a key.

        Parameters
        ----------
        *args : tuple
            A variable-length argument list where all elements except the last
            one are used to form the key (as a tuple) for accessing the
            dictionary. The last element of `args` is the name of the time
            series to set in the DataFrame.

        value : array-like
            The values to assign to the specified time series in the DataFrame.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If the specified key does not exist in `self.dict_df`.
        ValueError
            If the length of `value` does not match the length of the
            DataFrame's index.
        """
        dict_key = tuple(args[:-1])
        tsname = args[-1]
        self.dict_df[dict_key].loc[:, tsname] = value

    def keys(self):
        return self.dict_df.keys()

    def __getitem__(self, key):
        return self.dict_df[key]
    
    def __setitem__(self, key, value):
        self.dict_df[key] = value

    def __str__(self) -> str:
        result = []
        for key, df in self.dict_df.items():
            result.append(f"Key: {key}")
            result.append(str(df))
        return "\n".join(result)

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def _key2str(key):
        if isinstance(key, int):
            return str(key)
        return "_".join(map(str, key))

    @staticmethod
    def _str2key(key_str):
        if key_str.isdigit():
            return int(key_str)
        return tuple(map(int, key_str.split("_")))

    def write(self, prefix: str):
        """Write the DataFrames in `self.dict_df` to CSV files.

        Each DataFrame in `self.dict_df` is saved to a separate CSV file. The
        files are named using the provided `prefix` followed by the string
        representation of the key used in the dictionary.

        The key string pattern converts a tuple of integers (used as keys in the
        dictionary) into a string by joining the integers with underscores. For
        example, the tuple `(1, 2, 3)` becomes the string `"1_2_3"`. This string
        is then used in filenames to uniquely identify and relate the files to
        their corresponding dictionary keys.

        Parameters
        ----------
        prefix : str
            The prefix to use for naming the output files. Each file will be
            named as `prefix_key{key_str}.csv`, where `key_str` is the string
            representation of the dictionary key.

        Returns
        -------
        None

        Raises
        ------
        IOError
            If there is an issue writing the files to the disk.
        """
        for key, df in self.dict_df.items():
            key_str = self._key2str(key)
            df.to_csv(f"{prefix}_key{key_str}.csv", index=False)

    @classmethod
    def read(cls, prefix: str, format_ext="csv"):
        """Read DataFrames from files with the specified prefix and format, and
        reconstruct a `DataTimeSeries` object.

        This method searches for files with the given `prefix` and file
        extension (`format_ext`), reads them, and reconstructs the
        `DataTimeSeries` object with a dictionary where keys are derived from
        the filenames.

        The key string pattern converts a string back into the original tuple of
        integers by splitting the string on underscores. For example, the string
        `"1_2_3"` is converted back into the tuple `(1, 2, 3)`.

        Parameters
        ----------
        prefix : str
            The prefix used to identify the files to be read. The method expects
            files to follow the naming convention
            `prefix_key{key_str}.{format_ext}`.

        format_ext : str, optional
            The file format extension to look for (default is "csv").

        Returns
        -------
        DataTimeSeries
            A new instance of `DataTimeSeries` with the dictionary of DataFrames
            reconstructed from the files.

        Raises
        ------
        ValueError
            If no files with the specified prefix and format are found.

        IOError
            If there is an issue reading the files.
        """
        dict_df = {}
        file_pattern = f"{prefix}_key*.{format_ext}"
        files = glob.glob(file_pattern)

        if not files:
            raise ValueError(f"No files with prefix {prefix} found.")

        for file in files:
            # Extract key part from filename
            file_basename = os.path.basename(file)
            file_basename_noext = file_basename.rsplit("." + format_ext)[0]    
            key_str = file_basename_noext.rsplit("key")[1]

            key = cls._str2key(key_str)
            dict_df[key] = pd.read_csv(file)

        return cls(dict_df)

    @staticmethod
    def check_files_with_fullpath_prefix(fullpath_prefix):
        """Checks if there are files corresponding to the given full path
        prefix."""
        matching_files = glob.glob(fullpath_prefix + "*")
        return bool(matching_files)

@dataclass(slots=True)
class TSParameter:
    tsname: str
    default: float | int
    tsdata: DataTimeSeries = None

    def copy(self):
        return self.__class__(self.tsname, self.default)

    def set_tsdata(self, tsdata: Type[DataTimeSeries]):
        self.tsdata = tsdata

    def value(self, time: pd.Timestamp):
        return self.tsdata.get_value(self.tsname, time)

    def to_dict(self):
        return {"tsname": self.tsname, "default": self.default}

    def has_tsdata(self):
        return self.tsdata is not None

    @classmethod
    def from_dict(cls, data: Dict[str, Any] = None):
        if data is None:
            return None
        return cls(**data)

    def _validate_time_series(self):
        """Validate that self.tsname exists in the TimeSeriesManager."""
        if not self.tsdata.has_ts(self.tsname):
            pass
            # raise ValueError(f"Time series {self.tsname} does not exist in the
            # TimeSeriesManager.")

    def scale(self, factor: float, inplace=False):
        self.default *= factor
        if inplace:
            self.tsdata.ts[self.tsname] *= factor
        else:
            return self.tsdata.ts[self.tsname] * factor

    def __getitem__(self, key):
        return self.tsdata.get_value(self.tsname, *key)

    def __setitem__(self, key, value):
        self.tsdata.set_value(self.tsname, *key, value)

    def set_tsname(self, tsname: str):
        self.tsname = tsname


# ******************************************************************************#
# Power system components
# ******************************************************************************#
@dataclass
class InvData:
    cost_investment: float = None
    cost_maintenance: float = None


@dataclass(slots=True)
class Demand(Component):
    """Component representing a demand."""

    bus: int = None
    snom_MVA: float = None
    pnom_MW: float = None
    qnom_MVAr: float = None
    p_MW: TSParameter = None
    q_MVAr: TSParameter = None

    def scale(self, factor: float, inplace=False):
        if inplace:
            self._scale_inplace(factor)
            return self
        else:
            new_demand = self.copy()
            new_demand._scale_inplace(factor)

        return new_demand

    def _scale_inplace(self, factor: float):
        if self.pnom_MW is not None:
            self.pnom_MW *= factor
        if self.qnom_MVAr is not None:
            self.qnom_MVAr *= factor
        if self.snom_MVA is not None:
            self.snom_MVA *= factor

        if self.p_MW is not None and self.p_MW.has_tsdata():
            self.p_MW.scale(factor, inplace=True)

        if self.q_MVAr is not None and self.q_MVAr.has_tsdata():
            self.q_MVAr.scale(factor, inplace=True)


@dataclass(slots=True)
class PVGenerator(Component):
    """Component representing a generator."""

    bus: int = None
    snom_MVA: float = None
    pfmin: float = None
    pmax_MW: TSParameter = None

@dataclass(slots=True)
class PVSdmml(Component):
    Area: float = None
    Vmp: float = None
    Imp: float = None
    Voc: float = None
    Isc: float = None
    n_0: float = None
    mu_n: float = None
    N_series: float = None
    alpha_isc: float = None
    E_g: float = None
    R_shexp: float = None
    R_sh0: float = None
    R_shref: float = None
    R_s: float = None
    D2MuTau: float = None


@dataclass(slots=True)
class Bus(Component):
    """Component representing a bus."""

    vnom_kV: float = None
    bustype: int = None  # 0 slack, 1 PQ, 2 VTheta


@dataclass(slots=True)
class Branch(Component):
    """Component representing a branch."""

    busf: int = None
    bust: int = None
    r_pu: float = None
    x_pu: float = None
    b_pu: float = None
    snom_MVA: float = None


@dataclass
class InvBranch(Branch, InvData):
    pass


@dataclass
class InvGenerator(PVGenerator, InvData):
    pass


class Entity:
    _class_comp2type = {}
    _class_type2comp = {}
    _class_param2type = {}
    _class_type2param = {}

    @classmethod
    def register_serializable_component(cls, name: str, component_type):
        """Register a new component type."""
        cls._class_comp2type[name] = component_type
        cls._class_type2comp[component_type] = name

    @classmethod
    def register_serializable_parameter(cls, name: str, param_type):
        """Register a new parameter type."""
        cls._class_param2type[name] = param_type

    def __init_subclass__(cls, **kwargs):
        cls._class_comp2type = cls._class_comp2type.copy()
        cls._class_type2comp = cls._class_type2comp.copy()
        cls._class_param2type = cls._class_param2type.copy()
        cls._class_type2param = cls._class_type2param.copy()
        super().__init_subclass__(**kwargs)
        cls._initialize_class()

    @classmethod
    def _initialize_class(cls):
        """To be overridden by subclasses for custom initialization.

        This method is called when the class is first loaded, and it is used
        to register new components and parameters that are specific to the
        subclass.

        To register new components containers, use the following syntax:
        ```python
        cls.register_serializable_component("new_component", NewComponent)
        ```

        To register new parameters, use the following syntax:
        ```python
        cls.register_serializable_parameter("new_param", int)
        ```
        """
        pass

    def to_dict(self):
        data = {}
        component_names = self._class_comp2type.keys()
        components = [self.__getattribute__(name) for name in component_names]

        for component_name, components in zip(component_names, components):
            data[component_name] = {}
            for index, component in components.items():
                data[component_name][index] = component.to_dict()

        data["params"] = {
            k: self.__getattribute__(k) for k in self._class_param2type.keys()
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any] = None):
        params = data["params"]
        obj = cls(**params)
        for component_type, components in data.items():
            if component_type == "params":
                continue
            for index, component_data in components.items():
                component_cls = cls._class_comp2type[component_type]
                component = component_cls.from_dict(component_data)
                obj.add(component, index)
        return obj

    def __str__(self) -> str:
        """Pretty print self.to_dict()"""
        return json.dumps(self.to_dict(), indent=4)

    def __repr__(self) -> str:
        """Pretty print self.to_dict()"""
        return self.__str__()

    def _generate_index(self, component_type):
        """Generates a new index for a component type.

        If the index exists in the component type, it will generate a new one.
        """
        if not self.__getattribute__(self._class_type2comp[component_type]):
            return 0

        return (
            max(
                self.__getattribute__(
                    self._class_type2comp[component_type]
                ).keys()
            )
            + 1
        )

    def add(self, component: Component, index: int = None):

        if index is None:
            if component.index is None:
                index = self._generate_index(component.__class__)
                self._add_component(component, index)
                return index

            else:
                index = component.index

        components = self.__getattribute__(
            self._class_type2comp[component.__class__]
        )

        if index in components:
            raise ValueError(
                f"Index {index} obtained from component already exists in {component.__class__.__name__} dictionary."
            )

        self._add_component(component, index)
        return index

    def _add_component(self, component: Component, index: int):
        """Add a component to the system."""
        component.index = index
        self.__getattribute__(self._class_type2comp[component.__class__])[
            index
        ] = component

    def copy(self):
        data = self.to_dict()
        return self.from_dict(data)

    def components_iterator(self):
        for component_type, components in self._class_comp2type.items():
            for index, component in self.__getattribute__(
                self._class_type2comp[components]
            ).items():
                yield component

    def set_time_series(self, data: DataScenarioTime):
        for component in self.components_iterator():
            for param_name, param in component.__dict__.items():
                if isinstance(param, TSParameter):
                    param.set_tsdata(data)


class DataPowerSystem(Entity):

    @classmethod
    def _initialize_class(cls):
        cls.register_serializable_component("demands", Demand)
        cls.register_serializable_component("generators", PVGenerator)
        cls.register_serializable_component("buses", Bus)
        cls.register_serializable_component("branches", Branch)
        cls.register_serializable_component("inv_branches", InvBranch)
        cls.register_serializable_component("inv_generators", InvGenerator)

        cls.register_serializable_parameter("name", str)
        cls.register_serializable_parameter("index", int)
        cls.register_serializable_parameter("sbase_MVA", float)

    def __init__(
        self,
        name: str = None,
        index: int = None,
        sbase_MVA: float = None,
        demands: Dict[int, Demand] = None,
        generators: Dict[int, PVGenerator] = None,
        buses: Dict[int, Bus] = None,
        branches: Dict[int, Branch] = None,
        inv_branches: Dict[int, InvBranch] = None,
        inv_generators: Dict[int, InvGenerator] = None,
    ) -> None:

        # Parameters
        self.name = name
        self.index = index
        self.sbase_MVA = sbase_MVA

        # Components container
        self.demands = demands if demands is not None else {}
        self.generators = generators if generators is not None else {}
        self.buses = buses if buses is not None else {}
        self.branches = branches if branches is not None else {}
        self.inv_branches = inv_branches if inv_branches is not None else {}
        self.inv_generators = (
            inv_generators if inv_generators is not None else {}
        )










def case_ieee4bus():
    """CASE4_DIST Power flow data for 4 bus radial distribution system.

    See MATPOWER. Auxliar generators were removed and all non-slack buses
    converted to PQ buses.
    """

    bus_names = ["index", "bustype", "vnom_kV"]
    bus_vals = [[1, 0, 12.5], [2, 1, 12.5], [3, 1, 12.5], [4, 1, 12.5]]

    demand_names = ["bus", "snom_MVA", "pnom_MW", "qnom_MVAr"]
    demand_vals = [
        [2, 0.447, 0.4, 0.2],
        [3, 0.447, 0.4, 0.2],
        [4, 0.447, 0.4, 0.2],
    ]

    branches_names = ["busf", "bust", "r_pu", "x_pu", "b_pu", "snom_MVA"]
    branches_vals = [
        [2, 3, 0.003, 0.006, 0.0, 999],
        [1, 2, 0.003, 0.006, 0.0, 999],
        [4, 1, 0.003, 0.006, 0.0, 999],
    ]

    data = DataPowerSystem(sbase_MVA=1.0)

    for i in range(4):
        kwargs = {k: v for k, v in zip(bus_names, bus_vals[i])}
        data.add(Bus(**kwargs))

    for i in range(3):
        kwargs = {k: v for k, v in zip(demand_names, demand_vals[i])}
        kwargs["p_MW"] = TSParameter(f"p_MW{i}", kwargs["pnom_MW"])
        kwargs["q_MVAr"] = TSParameter(f"q_MVAr{i}", kwargs["qnom_MVAr"])
        data.add(Demand(**kwargs))

    for i in range(3):
        kwargs = {k: v for k, v in zip(branches_names, branches_vals[i])}
        data.add(Branch(**kwargs))

    # Change the resistance to reactance ratio to r/x = 2.7
    for i in range(3):
        data.branches[i].r_pu = 2.7 * data.branches[i].x_pu

    return data
