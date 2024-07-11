SNAM_V = 'V'
SNAM_VANG = 'ang'
SNAM_NSE = 'nse'
SNAM_BUSP = 'Pbus'
SNAM_BUSQ = 'Qbus'

# Line variables
SNAM_I = 'I'
SNAM_P = 'P'    # Active power flow into the line at bus_t in p.u.
SNAM_Q = 'Q'    # Reactive power flow into the line at bus_t in p.u.
SNAM_PF = 'PF'  # Active power flow into the line at bus_f in p.u.
SNAM_QF = 'QF'  # Reactive power flow into the line at bus_f in p.u.

SNAM_DGP = 'dgp'
SNAM_DGQ = 'dgq'
SNAM_DGPMAX = 'dgpmax'
PREFIX_LOAD = 'load'
SNAM_LOADP = PREFIX_LOAD + 'p'
SNAM_LOADQ = PREFIX_LOAD + 'q'
SNAM_LOADDGP = 'ldgp'
SNAM_LOADEVP = 'lev'
SNAM_B2BFP = 'b2bfp'
SNAM_B2BFQ = 'b2bfq'
SNAM_B2BTP = 'b2btp'
SNAM_B2BTQ = 'b2btq'

SNAM_LOADP_MW = 'loadp_mw'
SNAM_LOADQ_MVAR = 'loadq_mvar'

SNAM_LOADP_KW = 'loadp_kw'
SNAM_DGPMAX_KW = 'dgpmax_kw'

# Real time controller
SNAM_GPV = 'Gpv'
SNAM_GQV = 'Gqv'
SNAM_DLPP = 'DLpp'
SNAM_DLPQ = 'DLpq'
SNAM_DLQP = 'DLqp'
SNAM_DLQQ = 'DLqq'
SNAM_DPP = 'DPp'
SNAM_DPQ = 'DPq'

# Real time controller deltas
SNAM_DGDP = 'dgdp'
SNAM_DGDQ = 'dgdq'
SNAM_DGP_OPF = 'opfdgp'
SNAM_DGQ_OPF = 'opfdgq'
SNAM_V_OPF = 'opfV'
SNAM_V_DELTA = 'Vdelta'
SNAM_DGP_DELTA = 'dgpdelta'
SNAM_DGQ_DELTA = 'dgqdelta'
SNAM_V_NOM = 'nomV'
SNAM_VANG_NOM = 'nomVang'

# fobj names
SNAM_FOBJ_EXP = 'fobj2stage_exp'
SNAM_FOBJ_MAX = 'fobj2stage_max'
SNAM_FOBJ_CVAR = 'fobj2stage_cvar'

PARAMS = {
    'general': {'value': 'object'},
    'buses': {'vmin': 'float64', 'vmax': 'float64', 'vbase_kv': 'float64', 'loadp_mw': 'float64',
              'loadq_mvar': 'float64'},
    'branches': {'busf': 'int64', 'bust': 'int64', 'r': 'float64', 'x': 'float64', 'b': 'float64',
                 'imax': 'float64'},
    'loads': {'bus': 'int64'},
    'dgs': {'bus': 'int64', 'snom': 'float64'},
    'b2bs': {'busf': 'int64', 'bust': 'int64', 'snom': 'float64'},
    'trafos': {'busf': 'int64', 'bust': 'int64', 'r': 'float64', 'x': 'float64', 
                 'imax': 'float64', 'Ntaps': 'int32', 'Taps': 'float64', 'TransitionCosts': 'float64'},
    'caps': {'bus': 'int64', 'Ncaps': 'int32', 'Qstage':'float64', 'TransitionCosts': 'float64'},
}

PARAMS_TIME_CONFIG = {
    'tini': 'datetime',
    'tiniout': 'datetime',
    'tend': 'datetime',
    'dt': 'timedelta',
    'n_rh': 'int64'
}

PARAMS_CONFIG = {}
PARAMS_GENERAL = {
    'slack_bus': 'int64',
    'sbase_mva': 'float64',
    'cost_nse': 'float64',
    'cost_putility': 'float64',
    'cost_vlim': 'float64',
    'ctrl_active': 'bool',
    'ctrl_project': 'bool',
    'cost_losses': 'float64',
    'opf_model': 'string',
    'bilinear_approx': 'string',
    'ctrl_type': 'string',
    'cost_lasso_v': 'float64',
    'cost_lasso_x': 'float64',
    'polypol_deg': 'int64',
    'cost_stability': 'float64',
    'ctrl_robust_n_sce': 'int64',
    'ctrl_robust': 'bool',
    'ctrl_tunning_dg_mode': 'string',
    'mccormick_dv': 'float64',
    'mccormick_dg': 'float64',
    'cost_ctrl_change': 'float64',
    'scegen_type': 'string',
    'scegen_n_win_ahead': 'int64',
    'scegen_n_days_delay': 'int64',
    'scegen_n_days': 'int64',
    'scegen_n_win_before': 'int64',
    'risk_measure': 'string'
}


PARAMS_GENERAL_VALID = {
    'opf_model': {'lindistflow', 'socp', 'distflow', 'acopf'},
    'bilinear_approx': {'mccormick', 'neuman', 'bilinear'},
    'ctrl_type': {'droop', 'polypol', 'droop_polypol'},
    'ctrl_tunning_dg_mode': {'free', 'oid'},
    'scegen_type': {'ddus_naive', 'ddus_kmeans'},
    'risk_measure': {'expected_value', 'worst_case'}
}

PARAMS_PFIX = {
    'general': '_general.csv',
    'buses': '_buses.csv',
    'branches': '_branches.csv',
    'loads': '_loads.csv',
    'trafos': '_trafos.csv',
    'caps': '_caps.csv',
    'dgs': '_dgs.csv',
    'b2bs': '_b2bs.csv',
    'config': '_config.csv',
    'time_config': '_time_config.yaml',
    'data': '_data.csv',
    'results': '_results.csv',
}

PARAMS_METRICS = {
    'avg_losses_pu': 'float64',
    'avg_utility_injection_pu': 'float64',
    'avg_hnvv': 'float64',
    'avg_pv_curtailment_pu': 'float64',
    'vmax': 'float64',
    'vmin': 'float64'
}

SOLVER_OPTIONS_KNITRO = 'outlev=1 ms_enable=1 par_numthreads=4 outmode=0 ms_terminate=1' \
                        'feastol=2e-4'

PARAMS_INPUT_ANN = {
    'time':{'cos':'float64','sin':'float64'},
    'uncertain':{SNAM_LOADP: 'float64', SNAM_LOADQ: 'float64', SNAM_DGPMAX: 'float64'},
    'nominal_set_point':{SNAM_V: 'float64'},
    'nominal_control_actions':{SNAM_DGP:'float64',SNAM_DGQ:'float64'}
}

PARAMS_OUTPUT_ANN = {
    'FOBJ':'float64'
}