import numpy as np

####### simulation parameters #######
sim_number = 10
jobs_number = 500
name_sim ='thal_motor_'
update_function = 'version_normal'
integrator = 'forward_euler'
delta_t = 0.01
Ttau = 100
tau = 0.1

####### Network parameters #######
learning_rule = 'none'

# synaptic strength matrix of CS, CC, PV and SST neurons (rows are the presyn cell)
# Campognola 2022 PSP amplitude

w_initial = np.array([[0.010604,0.000000,0.074556,0.095266],
                    [0.000415,0.001789,0.011929,0.014098],
                    [-0.005575,-0.004927,-0.251221,-0.070342],
                    [-0.133411,-0.033353,-0.190775,-0.067977]])


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
input_pv_steady = [0]
input_sst_steady = [0]
input_cs_amplitude = [6,7,8,9,10]
input_cc_amplitude = [6,7,8,9,10]
input_pv_amplitude = [6,7,8,9,10]
input_sst_amplitude = [6,7,8,9,10]
spatialF = [1]
temporalF = [1]
spatialPhase = [1]

thal_degree = [0,1,2,3]
thal_scalars = [[1.0,1.0],[1.1,1.2],[1.3,1.4],[1.1,1.4],[1.3,1.6],[1.3,1.7]]
thal_prop = (np.array([0.44,0.22,0,0])*N).astype(int)