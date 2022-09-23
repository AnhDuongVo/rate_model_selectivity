from os.path import abspath
import sys
sys.path.append(abspath(''))
import numpy as np
from datetime import datetime
import time
import math
import importlib
import os

import Implementation.network_model as nm
from Implementation.helper import distributionInput, generate_connectivity, \
    calculate_selectivity, plot_activity
if len(sys.argv) != 0:
    p = importlib.import_module(sys.argv[1])
else:
    import test_config as p

np.random.seed(42)

def run_simulation(input_cs, input_cc, input_pv, input_sst,
                   spatialF,temporalF,spatialPhase,amplitude):

    # network parameters
    N = p.N
    prob = p.prob
    w_initial = p.w_initial
    w_noise = p.w_noise

    # prepare different orientation inputs
    degree = p.degree
    radians = []
    for i in degree:
        radians.append(math.radians(i))

    # Evaluation metrics
    nan_counter, not_eq_counter = 0, 0
    os_rel, ds_rel, ds_paper_rel = None, None, None
    os_mean_all, ds_mean_all, ds_paper_mean_all, n_rel_all = [], [], [], []

    ################## iterate through different initialisations ##################
    for sim in range(p.sim_number):
        title_folder = 'data/figures/sbi'+str(sim)
        if not(os.path.exists(title_folder)):
            os.mkdir(title_folder)

        # weights
        W_rec = generate_connectivity(N, prob, w_initial, w_noise)

        # eye matrix
        num_neurons = W_rec.shape[0]
        W_project_initial = np.eye(num_neurons)

        # initial activity
        initial_values = np.random.uniform(low=0, high=1, size=(sum(N),))

        activity_data = []
        success = 0

        ################## iterate through different inputs ##################
        for g in radians:
            # build network here
            Sn = nm.SimpleNetwork(W_rec,
                                  W_project=W_project_initial,
                                  nonlinearity_rule=p.nonlinearity_rule,
                                  integrator=p.integrator,
                                  delta_t=p.delta_t,
                                  tau=p.tau,
                                  Ttau=p.Ttau,
                                  update_function = p.update_function,
                                  learning_rule = p.learning_rule,
                                  gamma = p.gamma)

            # define inputs
            inputs = distributionInput(spatialF=spatialF,
                                       temporalF=temporalF,
                                       orientation=g,
                                       spatialPhase=spatialPhase,
                                       amplitude=amplitude,
                                       T=Sn.tsteps,
                                       steady=p.steady_input,
                                       input_cs = input_cs,
                                       input_cc = input_cc,
                                       input_pv = input_pv,
                                       input_sst = input_sst,
                                       N = N)

            # run
            activity, w = Sn.run(inputs, initial_values)
            activity = np.asarray(activity)

            # check nan
            if np.isnan(activity[-1]).all():
                nan_counter += 1
                break

            # check equilibrium
            a1 = activity[-3000:-1500, :]
            a2 = activity[-1500:, :]
            mean1 = np.mean(a1, axis=0)
            mean2 = np.mean(a2, axis=0)
            check_eq = np.sum(np.where(mean1 - mean2 < 0.05, np.zeros(np.sum(N)), 1))
            if check_eq > 0:
                not_eq_counter += 1
                print('not eq')
                #break
            else:
                print('eq')

            if g == radians[-1]:
                success = 1
            activity_data.append(activity)
        activity = np.array(activity_data)
        plot_activity(activity, N, title_folder)

        if success:
            activity_cs = np.mean(activity[:, -1500:, :N[0]], axis=1)
            activity_cc = np.mean(activity[:, -1500:, sum(N[:1]):sum(N[:2])], axis=1)
            activity_all = [activity_cs, activity_cc]

            # calculate proportion of reliable CS and CC cells (at least one active)
            n_rel = []
            for popu in range(2):
                for neuron in range(N[popu]):
                    n_on = 0
                    for stim in range(4):
                        if activity_all[popu][stim, neuron] > 0.0001:
                            n_on += 1
                n_rel.append(n_on/N[popu])
            n_rel_all.append(n_rel)

            os_mean, ds_mean, ds_paper_mean= calculate_selectivity_sbi(activity_all)
            os_mean_all.append(os_mean)
            ds_mean_all.append(ds_mean)
            ds_paper_mean_all.append(ds_paper_mean)

    # calculate mean of orientation and direction selectivity
    if os_mean_all != []:
        n_rel_data = np.mean(np.array(n_rel_all),axis=0)
        os_mean_data = np.mean(np.array(os_mean_all),axis=0)
        ds_mean_data = np.mean(np.array(ds_mean_all),axis=0)
        ds_paper_mean_data = np.mean(np.array(ds_paper_mean_all), axis=0)

        if (os_mean_data[0] - os_mean_data[1]) > 0.00001:
            os_rel = (os_mean_data[0] - os_mean_data[1])/(os_mean_data[0] + os_mean_data[1])
        if (ds_mean_data[0] - ds_mean_data[1]) > 0.00001:
            ds_rel = (ds_mean_data[0] - ds_mean_data[1]) / (ds_mean_data[0] + ds_mean_data[1])
        if (ds_paper_mean_data[0] - ds_paper_mean_data[1]) > 0.00001:
            ds_paper_rel = \
                (ds_paper_mean_data[0] - ds_paper_mean_data[1]) / (ds_paper_mean_data[0] + ds_paper_mean_data[1])
    else:
        n_rel_data, os_mean_data, ds_mean_data, ds_paper_mean_data = \
            [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]

    # collect results here
    row = []
    selectivity_data = [os_mean_data, ds_mean_data, ds_paper_mean_data, n_rel_data]
    for selectivity_data_i in selectivity_data:
        for d in selectivity_data_i:
            row.append(d)
    row = row + [os_rel,ds_rel,ds_paper_rel, nan_counter/p.sim_number, not_eq_counter/p.sim_number]
    return np.array(row)

############### prepare csv file ###############
for input_cs in p.input_cs:
    for input_cc in p.input_cc:
        for input_pv in p.input_pv:
            for input_sst in p.input_sst:
                for spatialF in p.spatialF:
                    for temporalF in p.temporalF:
                        for spatialPhase in p.spatialPhase:
                            for amplitude in p.amplitude:

                                print(run_simulation(input_cs, input_cc, input_pv, input_sst,
                                                spatialF,temporalF,spatialPhase,amplitude))