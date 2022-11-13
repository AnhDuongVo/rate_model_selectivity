import numpy as np
import math

####### simulation parameters #######
sim_number = 1
jobs_number = 4
name_sim ='timescales_'
update_function = 'version_normal'
integrator = 'forward_euler'
delta_t = 0.01
Ttau = 300

####### Network parameters #######
learning_rule = 'none'

# synaptic strength matrix of CS, CC, PV and SST neurons (rows are the presyn cell)
# Campognola 2022 PSP amplitude
w_visual = np.array([[0.27, 0, 1.01, 0.05],
                       [0.19, 0.24, 0.48, 0.09],
                       [-0.32, -0.52, -0.47, -0.44],
                       [-0.19, -0.11, -0.18, -0.19]])

w_auditory = np.array([[0.000965,0.000000,0.061790,0.118870],
                        [0.000154,0.000071,0.012358,0.021989],
                        [-0.161742,-0.142935,-0.194761,0.000000],
                        [-0.141726,-0.035431,-0.047393,0.000000]])

w_motor = np.array([[0.010604,0.000000,0.074556,0.095266],
                    [0.000415,0.001789,0.011929,0.014098],
                    [-0.005575,-0.004927,-0.251221,-0.070342],
                    [-0.133411,-0.033353,-0.190775,-0.067977]])

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
N = np.array([45, 275, 46, 34])

w_noise = 0.03 # synaptic weight noise

####### Activation function #######
nonlinearity_rule = 'supralinear'
gamma = 1

####### Input #######
degree = [0, 90, 180, 270]
radians = []
for i in degree:
    radians.append(math.radians(i))

input_cs_steady = [0]
input_cc_steady = [0]
input_pv_steady = [0]
input_sst_steady = [0]
input_cs_amplitude = [0,0.125,0.25,0.5,1,2,4]
input_cc_amplitude = [0,0.125,0.25,0.5,1,2,4]
input_pv_amplitude = [0,0.125,0.25,0.5,1,2,4]
input_sst_amplitude = [0,0.125,0.25,0.5,1,2,4]
input_cs_amplitude = [4]
input_cc_amplitude = [2]
input_pv_amplitude = [0]
input_sst_amplitude = [4]
tau_cs = [0.1,0.125,0.25,0.5,1,2,4]
tau_cc = [0.1,0.125,0.25,0.5,1,2,4]
tau_pv = [0.1,0.125,0.25,0.5,1,2,4]
tau_sst = [0.1,0.125,0.25,0.5,1,2,4]
tau = 0.1
spatialF = [1]
temporalF = [1]
spatialPhase = [1]
cc_cs_weight = [0.19]

