import numpy as np

####### simulation parameters #######
sim_number = 10
jobs_number = 200
name_sim ='thal_auditory_hyperparameter'
update_function = 'version_normal'
integrator = 'forward_euler'
delta_t = 0.01
Ttau = 100
tau = 0.1

####### Network parameters #######
learning_rule = 'none'

# synaptic strength matrix of CS, CC, PV and SST neurons (rows are the presyn cell)
# Campognola 2022 PSP amplitude

w_initial = np.array([[0.000965,0.000000,0.061790,0.118870],
                        [0.000154,0.000071,0.012358,0.021989],
                        [-0.161742,-0.142935,-0.194761,0.000000],
                        [-0.141726,-0.035431,-0.047393,0.000000]])


# Campognola 2022 PSP amplitude
# connection probability matrix of CS, CC, PV and SST neurons (rows are the presyn cell)

prob = np.array([[0.16, 0, 0.18, 0.23],
                  [0.09, 0.06, 0.22, 0.26],
                  [0.43, 0.38, 0.5, 0.14],
                  [0.52, 0.13, 0.29, 0.1]])


# number of CS, CC, PV and SST neurons
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
input_pv_steady = [1]
input_sst_steady = [1]
input_cs_amplitude = [6,7,8,9,10]
input_cc_amplitude = [0.1,0.5,1,2,3,4,5,6,7,8,9,10]
input_pv_amplitude = [0,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1,2,3,4,5]
input_sst_amplitude = [0,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1,2,3,4,5]
spatialF = [1]
temporalF = [1]
spatialPhase = [1]
weight_scalar = [0]
weight_choice = ['None']
#weight_scalar = [0,0.125,0.25,0.5,1,2,4,8]
#weight_choice = ['it_pt','pv_pt','pv_it','sst_pt','sst_it']


thal_degree = [0,1,2,3]
thal_scalars = [[1.0,1.0],[1.1,1.2],[1.3,1.4],[1.1,1.4],[1.3,1.6],[1.3,1.7]]
thal_prop = (np.array([0.44,0.22,0,0])*N).astype(int)