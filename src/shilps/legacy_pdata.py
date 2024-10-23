import numpy as np
import pandas as pd
import os
import networkx as nx
from shilps.legacy_errors import InputError
from shilps.legacy_global_definitions import *
from shilps.legacy_jpyaml import yaml
from datetime import datetime, timedelta

class Adn(object):
    def __init__(self):
        # dictionaries
        self.config = None
        self.time_config = dict.fromkeys(PARAMS_TIME_CONFIG.keys())
        self.grid_tables = dict.fromkeys(PARAMS.keys())
        for k, v in PARAMS.items():
            self.grid_tables[k] = pd.DataFrame(index=[], columns=v.keys())

        self.A_load = None
        self.A_dg = None
        self.A_b2bf = None
        self.A_b2bt = None
        self.A_brf = None
        self.A_brt = None
        self.A_trafost = None
        self.A_trafosf = None
        self.A_caps = None

        # graph
        self.__graph = None

        # time series dataframes
        self.df_data = None
        self.df_result = None

    def set(self, key, value):
        if self.grid_tables['general'] is None:
            self.grid_tables['general'] = pd.DataFrame(index=PARAMS_GENERAL.keys(),
                                                       columns=['value'])
        self.grid_tables['general'].loc[key, 'value'] = value

    def __getattr__(self, key):
        return self.grid_tables['general'].loc[key, 'value']

    @property
    def ybus(self):
        l_lines  = self.branches.index.to_list()
        l_buses  = self.buses.index.to_list()
        
        # l_buses.remove(self.slack_bus)
        l_trafos = self.trafos.index.to_list()
        print(l_trafos)

        A = pd.DataFrame(0., index=l_lines, columns=l_buses)
        ypr = pd.DataFrame(index=l_lines, columns=['ypr'])

        for i in l_lines:
            A.loc[i, self.branches.loc[i, 'busf']] = 1
            A.loc[i, self.branches.loc[i, 'bust']] = -1
            ypr.loc[i, 'ypr'] = 1 / complex(self.branches.loc[i, 'r'], self.branches.loc[i, 'x'])

        # for i in l_trafos: # añadir interacción del trafo

            

        return pd.DataFrame(np.transpose(A.to_numpy()) @ np.diag(np.ravel(ypr)) @ A.to_numpy(),
                            index=l_buses, columns=l_buses, dtype=complex)



    @property
    def tini(self):
        return self.time_config['tini']

    @property
    def tend(self):
        return self.time_config['tend']

    @property
    def dt(self):
        return self.time_config['dt']

    @property
    def tiniout(self):
        return self.time_config['tiniout']

    @property
    def buses(self):
        return self.grid_tables['buses']

    @property
    def branches(self):
        return self.grid_tables['branches']

    @property
    def trafos(self):
        dfx = self.grid_tables['trafos']
        dfx['Taps'] = [list(np.fromstring(k[1:-1], dtype=float, sep=' ')) for k in dfx['Taps'].to_list()]
        dfx['TransitionCosts'] = [list(np.fromstring(k[1:-1], dtype=float, sep=' ')) for k in dfx['TransitionCosts'].to_list()]
        return dfx

    @property
    def caps(self):
        dfx = self.grid_tables['caps']
        dfx['Qstage'] = [list(np.fromstring(k[1:-1], dtype=float, sep=' ')) for k in dfx['Qstage'].to_list()]
        dfx['TransitionCosts'] = [list(np.fromstring(k[1:-1], dtype=float, sep=' ')) for k in dfx['TransitionCosts'].to_list()]
        return self.grid_tables['caps']

    @property
    def b2bs(self):
        return self.grid_tables['b2bs']

    @property
    def dgs(self):
        return self.grid_tables['dgs']

    @property
    def loads(self):
        return self.grid_tables['loads']

    @property
    def l_loads(self):
        """
        :return: {list} The subset of buses that have connected loads
        """
        assert self.df_data is not None
        return [int(i.split(SNAM_LOADP)[-1]) for i in self.df_data.columns
                if i.startswith(SNAM_LOADP)]

    @property
    def l_buses0(self):
        return self.buses.index.to_list()

    @property
    def l_buses(self):
        l_buses = self.l_buses0
        l_buses.remove(self.slack_bus)
        return l_buses

    @property
    def l_branches(self):
        return self.branches.index.to_list()

    @property
    def l_trafos(self):
        return self.trafos.index.to_list()

    @property
    def l_caps(self):
        return self.caps.index.to_list()

    @property
    def l_dgs(self):
        return self.dgs.index.to_list()

    @property
    def l_b2bs(self):
        return self.b2bs.index.to_list()

    def make_connectivity_mats(self):
        # Initialization of containers
        b2bs = self.b2bs
        dgs = self.dgs
        branches = self.branches
        trafos = self.trafos
        caps = self.caps

        l_buses0 = self.l_buses0
        l_dgs = self.l_dgs
        l_branches = self.l_branches
        l_b2bs = self.l_b2bs
        l_loads = self.l_loads
        l_trafos = self.l_trafos
        l_caps = self.l_caps

        self.A_dg = pd.DataFrame(0, index=l_dgs, columns=l_buses0, dtype='float64')
        if l_branches:
            self.A_brf = pd.DataFrame(0, index=l_branches, columns=l_buses0, dtype='float64')
            self.A_brt = pd.DataFrame(0, index=l_branches, columns=l_buses0, dtype='float64')
        if l_b2bs:
            self.A_b2bf = pd.DataFrame(0, index=l_b2bs, columns=l_buses0, dtype='float64')
            self.A_b2bt = pd.DataFrame(0, index=l_b2bs, columns=l_buses0, dtype='float64')
        if l_loads:
            self.A_load = pd.DataFrame(0, index=l_loads, columns=l_buses0, dtype='float64')
        if l_trafos:
            self.A_trafosf = pd.DataFrame(0, index=l_trafos, columns=l_buses0, dtype='float64')
            self.A_trafost = pd.DataFrame(0, index=l_trafos, columns=l_buses0, dtype='float64')
        if l_caps:
            self.A_caps = pd.DataFrame(0, index=l_caps, columns=l_buses0, dtype='float64')

        for i in l_dgs:
            self.A_dg.loc[i, dgs.loc[i, 'bus']] = 1.

        for i in l_b2bs:
            self.A_b2bf.loc[i, b2bs.loc[i, 'busf']] = 1.
            self.A_b2bt.loc[i, b2bs.loc[i, 'bust']] = 1.

        for i in l_branches:
            self.A_brf.loc[i, branches.loc[i, 'busf']] = 1.
            self.A_brt.loc[i, branches.loc[i, 'bust']] = 1.

        for i in l_loads:
            self.A_load.loc[i, i] = 1.

        for i in l_trafos:
            self.A_trafosf.loc[i, trafos.loc[i, 'busf']] = 1.
            self.A_trafost.loc[i, trafos.loc[i, 'bust']] = 1.

        for i in l_caps:
            self.A_caps.loc[i, caps.loc[i, 'bus']] = 1.

    def time_map(self, tmap_mode=1):
        if tmap_mode == 1:
            time_map: pd.DatetimeIndex = pd.date_range(self.tini, self.tiniout - self.dt, freq=self.dt)
        elif tmap_mode == 2:
            time_map: pd.DatetimeIndex = pd.date_range(self.tiniout, self.tend, freq=self.dt)
        else:
            time_map: pd.DatetimeIndex = pd.date_range(self.tini, self.tend, freq=self.dt)
        return time_map

    def init_df_sol(self):
        idx_trange = pd.concat([self.time_map(1), self.time_map(2)], axis=0)

        l_buses0 = self.l_buses0
        l_branches = self.l_branches
        l_dgs = self.l_dgs
        l_loads = self.l_loads
        l_b2bs = self.l_b2bs

        map_names2sets = {
            SNAM_V: l_buses0,
            SNAM_VANG: l_buses0,
            SNAM_BUSP: l_buses0,
            SNAM_BUSQ: l_buses0,

            SNAM_I: l_branches,
            SNAM_P: l_branches,
            SNAM_Q: l_branches,
            SNAM_PF: l_branches,
            SNAM_QF: l_branches,

            SNAM_DGP: l_dgs,
            SNAM_DGQ: l_dgs,
            SNAM_DGPMAX: l_dgs,

            SNAM_LOADP: l_loads,
            SNAM_LOADQ: l_loads,
            SNAM_NSE: l_loads,

            SNAM_B2BFP: l_b2bs,
            SNAM_B2BFQ: l_b2bs,
            SNAM_B2BTP: l_b2bs,
            SNAM_B2BTQ: l_b2bs
        }
        col_names = []

        for k, v in map_names2sets.items():
            col_names += [k + str(i) for i in v]

        df_sol = pd.DataFrame(index=idx_trange, columns=col_names)

        return df_sol

    # FIXME: construct a normalizer function as matpower's ext2int()
    def relabel_branches(self):
        df_branches = self.branches
        df_aux_br = df_branches.loc[:, ['busf', 'bust']].copy()
        slack_bus = self.slack_bus

        map_lines2buses = pd.Series(index=df_branches.index.to_list(), dtype=int)
        l_nodes_to_visit = [slack_bus]
        while l_nodes_to_visit:
            i = l_nodes_to_visit.pop()
            aux = pd.concat(
                [df_aux_br.loc[df_aux_br['bust'] == i, 'busf'],
                 df_aux_br.loc[df_aux_br['busf'] == i, 'bust']],
                axis=0)
            l_nodes_to_visit += aux.to_list()

            map_lines2buses[aux.index] = aux.values
            df_aux_br.drop(aux.index, inplace=True)

        self.grid_tables['branches'].index = self.branches.index.map(map_lines2buses)
        swap_ids = self.branches.index == self.branches['bust']
        self.grid_tables['branches'].loc[swap_ids, ['busf', 'bust']] = \
            self.grid_tables['branches'].loc[swap_ids, ['bust', 'busf']].values
        self.grid_tables['branches'][['busf', 'bust']] = self.grid_tables['branches'][
            ['busf', 'bust']].astype(int)


    @property
    def graph(self):
        if self.__graph is None:
            df_branches = self.grid_tables['branches']
            df_buses = self.grid_tables['buses']
            df_trafos = self.grid_tables['trafos']
            assert df_branches is not None
            assert df_buses is not None
            assert df_trafos is not None
            
            self.__graph = nx.Graph()
            self.__graph.add_nodes_from({k:k for k in df_buses.index.to_list()})
            self.__graph.add_edges_from(list(zip(df_branches['busf'].to_list(), df_branches['bust'].to_list())), color='black', label = "L")
            self.__graph.add_edges_from(list(zip(df_trafos['busf'].to_list(), df_trafos['bust'].to_list())), color='r', label = "T")

        return self.__graph

    def grid_pointer(self):
        ret = Adn()
        ret.grid_tables = self.grid_tables
        return ret

    def read(self, fol_name, name):
        fn = os.path.join(fol_name, name)
        l_related_files = [i for i in os.listdir(fol_name) if i.startswith(name)]

        for k in self.grid_tables.keys():
            if name + PARAMS_PFIX[k] in l_related_files:
                self.grid_tables[k] = pd.read_csv(fn + PARAMS_PFIX[k], sep='\t', index_col='index')
        self.grid_tables['general']['value'] = self.grid_tables['general']['value'].astype('object')
        local_fn_data = name + PARAMS_PFIX['data']
        if local_fn_data in l_related_files:
            self.df_data = pd.read_csv(os.path.join(fol_name, local_fn_data), sep='\t',
                                       index_col='time')
            self.df_data.index = pd.to_datetime(self.df_data.index)

        local_fn_timeconfig = name + PARAMS_PFIX['time_config']
        if local_fn_timeconfig in l_related_files:
            fn_timeconfig = os.path.join(fol_name, local_fn_timeconfig)
            # Yaml read time_config file
            with open(fn_timeconfig, 'r') as hfile:
                time_config = yaml.load(hfile, yaml.FullLoader)
            self.time_config = time_config
        # Casting general table
        self.__cast_general_table()

        # Infer df_data freq
        freq = pd.infer_freq(self.df_data.index)
        if freq is not None:
            self.df_data = self.df_data.asfreq(freq)

        # Validation
        self.validate()

    def read_config(self, file_name, append=True):
        df_input_config = pd.read_csv(file_name, sep='\t', index_col='index')
        if append:
            # OPTI: There must be a more appropriate way to merge dataframes
            set_input = set(df_input_config.index.to_list())
            set_existing = set(self.grid_tables['general'].index.to_list())
            idx_new = set_existing.union(set_input)
            df_new = pd.DataFrame(index=idx_new, columns=self.grid_tables['general'].columns)
            df_new.update(self.grid_tables['general'])
            df_new.update(df_input_config)
            self.grid_tables['general'] = df_new
        else:
            self.grid_tables['general'] = df_input_config
        self.__cast_general_table()

    def __cast_general_table(self):
        # TODO2: Casting booleans gives True all the time
        for i in self.grid_tables['general'].index:
            type_instance = PARAMS_GENERAL[i]
            if type_instance == 'int64':
                if isinstance(self.grid_tables['general'].loc[i, 'value'], str):
                    self.grid_tables['general'].loc['slack_bus', 'value'] = (
                        int(float(self.grid_tables['general'].loc['slack_bus', 'value']))
                    )
            self.grid_tables['general'].loc[i, :] = (
                self.grid_tables['general'].loc[i, :].astype(type_instance)
            )

    def write(self, fol_name, name):
        fn = os.path.join(fol_name, name)
        for k, v in self.grid_tables.items():
            if v is not None:
                v.to_csv(fn + PARAMS_PFIX[k], sep='\t', index_label='index')
        fn_data = fn + PARAMS_PFIX['data']
        if self.df_data is not None:
            self.df_data.to_csv(fn_data, sep='\t', index_label='time')

    def validate(self):
        # Validating general table
        for key_param in PARAMS_GENERAL_VALID.keys():
            if key_param in self.grid_tables['general'].index:
                if not (self.grid_tables['general'].loc[key_param, 'value'] in
                        PARAMS_GENERAL_VALID[key_param]):
                    raise InputError('Invalid config param value for {}. Use any in {}'.format(
                        key_param, PARAMS_GENERAL_VALID[key_param]))

        # Validate types time_config
        dict_types = {'datetime': datetime, 'timedelta': timedelta, 'int64': int}
        for k, v in self.time_config.items():
            if v is not None:
                expected_type = dict_types[PARAMS_TIME_CONFIG[k]]
                if not isinstance(v, expected_type):
                    raise InputError('Type for parameter {} is {} and should be {}'.format(
                        k, type(v), expected_type
                    ))

    def se_distance_to_root(self):
        slack_bus = self.slack_bus
        assert slack_bus == 0
        l_buses = self.l_buses
        se_n = pd.Series(index=l_buses, dtype='int64')
        for i in l_buses:
            n = 0
            j = i
            while j != slack_bus:
                j = int(self.branches['bust'][self.branches.loc[:, 'busf'] == j].iloc[0])
                n += 1
            se_n[i] = n
        return se_n

