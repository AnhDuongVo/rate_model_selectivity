from os.path import abspath
import sys
import numpy as np
import math
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import importlib
from datetime import datetime
import time
import csv
from joblib import Parallel, delayed
import Implementation.network_model as nm
from Implementation.helper import distributionInput_thalamus, generate_connectivity, \
    calculate_selectivity_test,plot_activity,get_thal_scalar

if len(sys.argv) != 0:
    p = importlib.import_module(sys.argv[1])
else:
    import configs.test_config_thal as p

# remove top and right axis from plots
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
sys.path.append(abspath(''))

def run_simulation(input_cs_steady, input_cc_steady, input_pv_steady, input_sst_steady, input_cs_amplitude,
                   input_cc_amplitude, input_pv_amplitude, input_sst_amplitude,
                   weight_scalar, weight_choice,
                   spatialF, temporalF, spatialPhase,
                   thal_degree,thal_scalars,start_time,title):
    if not(thal_scalars == [1.0,1.0] and thal_degree in [1,2,3]):
        # network parameters
        N = p.N
        prob = p.prob
        w_initial = p.w_initial
        if weight_choice == 'it_pt':
            w_initial[1][0] *= weight_scalar
        elif weight_choice == 'pv_pt':
            w_initial[2][0] *= weight_scalar
        elif weight_choice == 'pv_it':
            w_initial[2][1] *= weight_scalar
        elif weight_choice == 'sst_pt':
            w_initial[3][0] *= weight_scalar
        elif weight_choice == 'sst_it':
            w_initial[3][1] *= weight_scalar
        w_noise = p.w_noise

        # input parameters
        amplitude = [input_cs_amplitude, input_cc_amplitude, input_pv_amplitude, input_sst_amplitude]
        steady_input = [input_cs_steady, input_cc_steady, input_pv_steady, input_sst_steady]

        # prepare different orientation inputs
        degree = p.degree
        radians = []
        for i in degree:
            radians.append(math.radians(i))

        # Evaluation metrics
        nan_counter, not_eq_counter = 0, 0
        os_rel, ds_rel, os_paper_rel = 0, 0, 0
        os_mean_all, ds_mean_all, os_paper_mean_all, n_rel_all, a_mean_all, a_std_all = [], [], [], [], [], []

        thal_scalar_matrix = get_thal_scalar(thal_degree, thal_scalars)

        ################## iterate through different initialisations ##################
        for sim in range(p.sim_number):
            #print('Sim')
            #title_folder = 'data/figures/sbi' + str(sim)
            #if not (os.path.exists(title_folder)):
            #    os.mkdir(title_folder)

            # weights
            W_rec = generate_connectivity(N, prob, w_initial, w_noise)
            W_rec = W_rec / max(np.linalg.eigvals(W_rec).real)

            # eye matrix
            num_neurons = W_rec.shape[0]
            W_project_initial = np.eye(num_neurons)

            # initial activity
            # initial_values = np.random.uniform(low=0, high=1, size=(sum(N),))
            initial_values = np.zeros((sum(N),))
            activity_data = []
            success = 0
            length = np.random.uniform(0, 1, (np.sum(N),))
            angle = np.pi * np.random.uniform(0, 2, (np.sum(N),))
            a_data = np.sqrt(length) * np.cos(angle)
            b_data = np.sqrt(length) * np.sin(angle)

            ################## iterate through different inputs ##################
            for g in range(len(radians)):
                # build network here
                Sn = nm.SimpleNetwork(W_rec, W_project=W_project_initial, nonlinearity_rule=p.nonlinearity_rule,
                                      integrator=p.integrator, delta_t=p.delta_t, tau=p.tau, Ttau=p.Ttau,
                                      update_function=p.update_function, learning_rule=p.learning_rule,
                                      gamma=p.gamma)

                # define inputs
                inputs = distributionInput_thalamus(a_data=a_data, b_data=b_data, spatialF=spatialF,
                                                    temporalF=temporalF, orientation=radians[g], spatialPhase=spatialPhase,
                                                    amplitude=amplitude, T=Sn.tsteps, steady_input=steady_input, N=N,
                                                    thal_scalar=thal_scalar_matrix[g], thal_prop=p.thal_prop)
                # run
                activity = Sn.run(inputs, initial_values)
                activity = np.asarray(activity)

                # check nan
                if np.isnan(activity[-1]).all():
                    nan_counter += 1
                    break

                # check equilibrium
                a1 = activity[-2000:-1000, :]
                a2 = activity[-1000:, :]
                mean1 = np.mean(a1, axis=0)
                mean2 = np.mean(a2, axis=0)
                check_eq = np.sum(np.where(mean1 - mean2 < 0.05, np.zeros(np.sum(N)), 1))
                if check_eq > 0:
                    not_eq_counter += 1
                    break
                if radians[g] == radians[-1]:
                    success = 1
                activity_data.append(activity)
            activity = np.array(activity_data)
            # plot_activity(activity, N, title_folder, sim)

            if success:
                a_mean = [np.mean(activity[:, -1500:, :N[0]]),
                          np.mean(activity[:, -1500:, sum(N[:1]):sum(N[:2])]),
                          np.mean(activity[:, -1500:, sum(N[:2]):sum(N[:3])]),
                          np.mean(activity[:, -1500:, sum(N[:3]):sum(N)])]
                a_std = [np.std(activity[:, -1500:, :N[0]]),
                         np.std(activity[:, -1500:, sum(N[:1]):sum(N[:2])]),
                         np.std(activity[:, -1500:, sum(N[:2]):sum(N[:3])]),
                         np.std(activity[:, -1500:, sum(N[:3]):sum(N)])]
                a_mean_all.append(a_mean)
                a_std_all.append(a_std)

                activity_cs = np.mean(activity[:, -1500:, :N[0]], axis=1)
                activity_cc = np.mean(activity[:, -1500:, sum(N[:1]):sum(N[:2])], axis=1)
                activity_pv = np.mean(activity[:, -1500:, sum(N[:2]):sum(N[:3])], axis=1)
                activity_sst = np.mean(activity[:, -1500:, sum(N[:3]):sum(N[:4])], axis=1)
                activity_all = [activity_cs, activity_cc, activity_pv, activity_sst]

                """
                fig, axs = plt.subplots(1, 4, figsize=(20, 4))
                for figure_i in range(4):
                    axs[figure_i].bar(range(activity_cc.shape[1]), activity_cc[figure_i])
                plt.title('CC Stim')
                plt.show()
    
                fig, axs = plt.subplots(1, 4, figsize=(20, 4))
                for figure_i in range(4):
                    axs[figure_i].bar(range(activity_cs.shape[1]), activity_cs[figure_i])
                plt.title('CS Stim')
                plt.show()"""

                # calculate proportion of reliable CS and CC cells (at least one active)
                n_rel = []
                for popu in range(4):
                    n_on = 0
                    for neuron in range(N[popu]):
                        n_add = 0
                        for stim in range(4):
                            if activity_all[popu][stim, neuron] > 0.0001:
                                n_add = 1
                        n_on += n_add
                    n_rel.append(n_on / N[popu])
                n_rel_all.append(n_rel)

                os_mean, ds_mean, os_paper_mean = calculate_selectivity_test(activity_all)
                os_mean_all.append(os_mean)
                ds_mean_all.append(ds_mean)
                os_paper_mean_all.append(os_paper_mean)

        # calculate mean of orientation and direction selectivity
        if os_mean_all != []:
            n_rel_data = np.mean(np.array(n_rel_all), axis=0)
            os_mean_data = np.mean(np.array(os_mean_all), axis=0)
            ds_mean_data = np.mean(np.array(ds_mean_all), axis=0)
            os_paper_mean_data = np.mean(np.array(os_paper_mean_all), axis=0)

            if np.abs((os_mean_data[0] - os_mean_data[1])) > 0.00001:
                os_rel = (os_mean_data[0] - os_mean_data[1]) / (os_mean_data[0] + os_mean_data[1])
            if np.abs((ds_mean_data[0] - ds_mean_data[1])) > 0.00001:
                ds_rel = (ds_mean_data[0] - ds_mean_data[1]) / (ds_mean_data[0] + ds_mean_data[1])
            if np.abs((os_paper_mean_data[0] - os_paper_mean_data[1])) > 0.00001:
                os_paper_rel = \
                    (os_paper_mean_data[0] - os_paper_mean_data[1]) / (os_paper_mean_data[0] + os_paper_mean_data[1])
        else:
            n_rel_data, os_mean_data, ds_mean_data, os_paper_mean_data = \
                [math.nan, math.nan], [math.nan, math.nan], [math.nan, math.nan], [math.nan, math.nan]

        a_mean_data = np.mean(np.array(a_mean_all), axis=0)
        a_std_data = np.std(np.array(a_mean_all), axis=0)

        # collect results here
        row = [input_cs_amplitude, input_cc_amplitude, input_pv_amplitude, input_sst_amplitude,
               thal_degree, thal_scalars,
               weight_scalar, weight_choice,
               nan_counter,not_eq_counter]
        selectivity_data = [os_mean_data, ds_mean_data, os_paper_mean_data, a_mean_data, a_std_data]
        for selectivity_data_i in selectivity_data:
            for d in selectivity_data_i:
                row.append(d)
        row = row + [os_rel, ds_rel, os_paper_rel,n_rel_data,time.time() - start_time]

        # write into csv file
        with open(title, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        return(row)

############### prepare csv file ###############

now = datetime.now() # current date and time
time_id = now.strftime("%m:%d:%Y_%H:%M:%S")
title = 'data/' + p.name_sim + time_id + '.csv'

row = ['cs_amplitude', 'cc_amplitude', 'pv_amplitude', 'sst_amplitude',
       'thal_degree','thal_scalars',
       'weight_scalar','weight_choice',
        'nan_counter','not_eq_counter',
        'os_mean1','os_mean2','os_mean3','os_mean4',
        'ds_mean1','ds_mean2','ds_mean3','ds_mean4',
        'os_paper_mean1','os_paper_mean2','os_paper_mean3','os_paper_mean4',
        'a_mean1','a_mean2','a_mean3','a_mean4',
        'a_std1','a_std2','a_std3','a_std4',
        'os rel','ds rel','os_paper_rel','n_rel_data',
        'time']

f = open(title, 'w')
writer = csv.writer(f)
writer.writerow(row)
f.close()

############### start simulation ###############

start_time = time.time()

# use joblib to parallelize simulations with different parameter values

Parallel(n_jobs=p.jobs_number)(delayed(run_simulation)(input_cs_steady, input_cc_steady, input_pv_steady,
                                                       input_sst_steady, input_cs_amplitude, input_cc_amplitude,
                                                       input_pv_amplitude, input_sst_amplitude,
                                                       weight_scalar, weight_choice,
                                                       spatialF, temporalF,spatialPhase,
                                                       thal_degree,thal_scalars,start_time,title)
                    for input_cs_steady in p.input_cs_steady
                    for input_cc_steady in p.input_cc_steady
                    for input_pv_steady in p.input_pv_steady
                    for input_sst_steady in p.input_sst_steady
                    for input_cs_amplitude in p.input_cs_amplitude
                    for input_cc_amplitude in p.input_cc_amplitude
                    for input_pv_amplitude in p.input_pv_amplitude
                    for input_sst_amplitude in p.input_sst_amplitude
                    for thal_degree in p.thal_degree
                    for thal_scalars in p.thal_scalars
                    for weight_scalar in p.weight_scalar
                    for weight_choice in p.weight_choice
                    for spatialF in p.spatialF
                    for temporalF in p.temporalF
                    for spatialPhase in p.spatialPhase)