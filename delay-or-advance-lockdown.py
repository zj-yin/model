#!/usr/bin/env python
# coding: utf-8


get_ipython().run_line_magic('matplotlib', 'inline')


import numpy as np
import pandas as pd
import pickle as pkl
from itertools import product
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
from matplotlib import pyplot as plt
from datetime import timedelta, datetime
from joblib import delayed, Parallel
from copy import copy

from core import  do_simulation
from helpers import Params, T, get_T1_and_T2, R0, plot_total, DATE_FORMAT, total_to_csv, save_to_json, save_bundle, makedir_if_not_there

from const import STATE, COLORS, NUM_STATES,  STATES

import os 





p0_time = pkl.load(open('output/p0_time.pkl', 'rb'))
lockdown_time = T('2020/3/21')




params_before = pkl.load(
    open('output/params_before_lockdown.pkl', 'rb')
)

'''
bed_info = pkl.load(open('data/bed_info.pkl', 'rb'))

params_after = pkl.load(
    open('output/params_after_lockdown.pkl', 'rb')
)

'''




total_days =  360




def one_run(params_after,bed_info,delta_t,total_days=360):
    delta = timedelta(days=delta_t)
    assumed_ld_date = lockdown_time + delta
    days_to_p0 = (assumed_ld_date - p0_time).days
    print('lockdown date',  assumed_ld_date)
    params = Params(
        total_population=params_before.total_population,
        initial_num_E=params_before.initial_num_E,
        initial_num_I=params_before.initial_num_I,
        initial_num_M=params_before.initial_num_M,    
        alpha=[(0, params_before.alpha), (days_to_p0, params_after.alpha)],
        beta=[(0, params_before.beta), (days_to_p0, params_after.beta)],
        mu_ei=params_after.mu_ei,
        mu_mo=params_after.mu_mo,
        k_days=params_after.k_days,
        x0_pt=params_after.x0_pt,          
        stages=[days_to_p0]
    )
    
    total, delta, increase, trans, stats = do_simulation(total_days, bed_info, params, p0_time=p0_time)
    bundle = [total, delta, increase, trans]
    return assumed_ld_date, delta_t, bundle, stats



file_path_factor='output/params_after_lockdown/'
file_list_factor=os.listdir(file_path_factor)
file_path_bed='data/bed_info/'
file_list_bed=os.listdir(file_path_bed)
for f_factor in file_list_factor:
    params_after=pkl.load(open(file_path_factor+f_factor, 'rb'))
    factor_one=f_factor.split(".pkl")[0].split("_")[3]
    for f_bed in file_list_bed:
        bed_info=pkl.load(open(file_path_bed+f_bed, 'rb'))
        bed_one=f_bed.split(".pkl")[0].split("_")[2]
        rows = Parallel(n_jobs=1)(delayed(one_run)(params_after,bed_info,delta_t, total_days) for  delta_t in range(-7, 8))
        makedir_if_not_there('figs/advance-or-delay-lockdown/')
        for dt, days, bundle, stats in rows:
            dt_str = dt.strftime('%y-%m-%d')
            print(days)
            print(dt_str)
            fig, ax = plot_total(bundle[0], p0_time, total_days)
            fig.savefig(f'figs/advance-or-delay-lockdown/factor-{factor_one}-bed-{bed_one}-{dt_str}({days}).pdf')
            save_bundle(bundle, p0_time, total_days, f'output/tbl/advance-or-delay-lockdown/factor-{factor_one}-bed-{bed_one}-{dt_str}-({days})/')
            save_to_json(stats, f'output/tbl/advance-or-delay-lockdown/factor-{factor_one}-bed-{bed_one}-{dt_str}-({days})/stats.txt')

