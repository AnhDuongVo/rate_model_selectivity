import numpy as np

####### simulation parameters #######
sim_number = 10
jobs_number = 300
name_sim ='thal_visual_hyperparameter_three_sim'
update_function = 'version_normal'
integrator = 'forward_euler'
delta_t = 0.01
Ttau = 100
tau = 0.1

####### Network parameters #######
learning_rule = 'none'

# synaptic strength matrix of CS, CC, PV and SST neurons (rows are the presyn cell)
# Campognola 2022 PSP amplitude

w_initial = np.array([[0.27, 0, 1.01, 0.05],
                       [0.19, 0.24, 0.48, 0.09],
                       [-0.32, -0.52, -0.47, -0.44],
                       [-0.19, -0.11, -0.18, -0.19]])


# Campognola 2022 PSP amplitude
# connection probability matrix of CS, CC, PV and SST neurons (rows are the presyn cell)

prob = np.array([[0.16, 0, 0.18, 0.23],
                  [0.09, 0.06, 0.22, 0.26],
                  [0.43, 0.38, 0.5, 0.14],
                  [0.52, 0.13, 0.29, 0.1]])


# number of CS, CC, PV and SST neurons
#N = np.array([90, 550, 92, 68])
N = np.array([45, 275, 46, 34])

w_noise = 0.03 # synaptic weight noise

####### Activation function #######
nonlinearity_rule = 'supralinear'
gamma = 1

####### Input #######
# [0.0625,0.125,0.25,0.5,1,2,4,8,16]
degree = [0, 90, 180, 270]

input_cs_steady = [0]
input_cc_steady = [0]
input_pv_steady = [0]
input_sst_steady = [0]
input_cs_amplitude = [6,7,8,9,10]
input_cc_amplitude = [4,5,6,7,8,9,10]
input_pv_amplitude = [1,2,3,4,5,6]
input_sst_amplitude = [1,2,3,4,5,6]
spatialF = [10]
temporalF = [50]
spatialPhase = [1]
weight_scalar = [1,0]
weight_choice = ['it_pt'] #['it_pt','pv_pt','pv_it','sst_pt','sst_it']

thal_degree = [3]
#thal_degree = [0]
thal_scalars = [[1.0,1.0],[1.1,1.2],[1.3,1.6],[1.3,1.7]]
#thal_scalars = [[1.0,1.0]]
thal_prop = (np.array([0.35,0.15,0,0])*N).astype(int)