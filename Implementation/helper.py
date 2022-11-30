import numpy as np
np.random.seed(42)
import math
import matplotlib as mpl
import matplotlib.pyplot as plt

# remove top and right axis from plots
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False


def distributionInput(a_data,b_data,spatialF, temporalF, orientation, spatialPhase, amplitude, T, steady_input, N):
    """
    Generates a moving bar as input to CS, CC, PV, SST. 
    Using the function from textbook theoretical neurosciences. 
    Turning the image into stimulus by converting the difference between that pixel over time and the 
    difference between the pixel and the overall backaground level of luminance. 
    Output: Stimulus to the L5 neuron
    Steady_input: making a check to make sure the input is steady.
    """
    
    i = 0
    inputs_p_all = []
    N_indices = [[0, N[0]], [sum(N[:1]), sum(N[:2])], [sum(N[:2]), sum(N[:3])], [sum(N[:3]), sum(N)]]
    for popu in N_indices:
        inputs_p = []

        if steady_input[i] > 0.5:
            for t in range(T):
                inputs_p.append(amplitude[i] * np.cos(
                    spatialF * a_data[popu[0]:popu[1]] * np.cos(orientation) +
                    spatialF * b_data[popu[0]:popu[1]] * np.sin(orientation) - spatialPhase)
                                       * np.cos(temporalF) + amplitude[i])
            inputs_p = np.array(inputs_p)
        else:
            for t in range(T):
                inputs_p.append(amplitude[i] * np.cos(
                    spatialF * a_data[popu[0]:popu[1]] * np.cos(orientation) +
                    spatialF * b_data[popu[0]:popu[1]] * np.sin(orientation) - spatialPhase)
                                       * np.cos(temporalF * t) + amplitude[i])
            inputs_p = np.array(inputs_p)
        i += 1
        inputs_p_all.append(inputs_p)

    inputs = np.concatenate((inputs_p_all), axis=1)

    return (inputs)

def distributionInput_thalamus(a_data,b_data,spatialF, temporalF, orientation, spatialPhase,
                               amplitude, T, steady_input, N, thal_scalar, thal_prop):
    """
    Generates a moving bar as input to CS, CC, PV, SST.

    """

    i = 0
    inputs_p_all = []
    N_indices = [[0, N[0]], [sum(N[:1]), sum(N[:2])], [sum(N[:2]), sum(N[:3])], [sum(N[:3]), sum(N)]]
    for popu in N_indices:
        inputs_p = []

        if steady_input[i] > 0.5:
            inputs_p = np.ones((T,N[i]))*amplitude[i]

            # thalamus input
            inputs_p[:,:thal_prop[i]] *= thal_scalar[i]
            """
            for t in range(T):
                input_item = amplitude[i] * np.cos(
                    spatialF * a_data[popu[0]:popu[1]] * np.cos(orientation) +
                    spatialF * b_data[popu[0]:popu[1]] * np.sin(orientation) - spatialPhase)*np.cos(temporalF) + amplitude[i]

                # thalamus input
                input_item[:thal_prop[i]] *= thal_scalar[i]

                inputs_p.append(input_item)"""
        else:
            for t in range(T):
                input_item = amplitude[i] * np.cos(
                    spatialF * a_data[popu[0]:popu[1]] * np.cos(orientation) +
                    spatialF * b_data[popu[0]:popu[1]] * np.sin(orientation) - spatialPhase)* np.cos(temporalF * t) + amplitude[i]

                # thalamus input
                input_item[:thal_prop[i]] *= thal_scalar[i]

                inputs_p.append(input_item)

        inputs_p_all.append(np.array(inputs_p))
        i += 1

    inputs = np.concatenate((inputs_p_all), axis=1)

    return (inputs)

def get_thal_scalar(degree=0,scalars=[1.1,1.3]):
    thal_scalar = np.ones((4,4))
    thal_scalar[degree][:2] = scalars[1]
    thal_scalar[(degree+1)%4][:2] = scalars[0]
    thal_scalar[(degree+3)%4][:2] = scalars[0]
    return(thal_scalar)

def create_synapses(N_pre, N_post, c, same_population=False):
    """
    Create random connections between two groups or within one group.
    :param N_pre: number of neurons in pre group
    :param N_post: number of neurons in post group
    :param c:   connectivity (probability for connecting, synaptic strength)
    :param same_population: whether to allow autapses (connection of neuron to itself if pre = post population)
    :return: 2xN_con array of connection indices (pre-post pairs)
    """

    # indegree: the expectation of how many pre neurons will form synapses with post neurons
    indegree = int(np.round(c * N_pre))  # no variance of indegree = fixed indegree

    i = np.array([], dtype=int)
    j = np.array([], dtype=int)

    for n in range(N_post):

        if same_population:  
            # if autapses are disabled, remove index of present post neuron from pre options
            opts = np.delete(np.arange(N_pre, dtype=int), n)
        else:
            opts = np.arange(N_pre, dtype=int)

        # Randomly select the expected number of neurons from pre to connect with post neurons
        pre = np.random.choice(opts, indegree, replace=False)

        # add connection indices to list pre -> indexes for pre-neurons; post -> index of the post neuron
        i = np.hstack((i, pre))  
        j = np.hstack((j, np.repeat(n, indegree)))

    # return a 2x(N_post*indegree) array, representing the connection between each pair of pre-post neurons
    return np.array([i, j])

def normal_distr_weights(weight, w_noise):
    """
    Generate weights with normally distributed weight noise.
    weight: Experimental tested weight in the literature. The target 
    situation that we want to achieve.
    """

    if weight > 0:
        weight = np.abs(np.random.normal(weight, w_noise))
    elif weight < 0:
        weight = -np.abs(np.random.normal(weight, w_noise))
    return (weight)


def generate_connectivity(N, p, w_initial, w_noise):
    """
    Generates a connectivity matrix where rows are postsyn. neuron and columns are presynaptic neuron.
    p: Probability matrix 4x4 (connection probability matrix of CS, CC, PV and SST neurons (rows are the presyn cell))
    w_initial: the target weight to achieve according to Campognola 2022 PSP amplitude. Presented 
               in terms of synaptic strength matrix (sequared, 4x4) where rows are presynaptic cells
    w_noise: degree of noise introduced to the synpatic connection
    N: 1x4 vector representing the number of CS, CC, PV and SST neurons
    """
    # Individual neuron matrix
    W_rec = np.zeros((np.sum(N), np.sum(N)))

    # iterate through all cell population
    for pre in range(p.shape[0]):
        for post in range(p.shape[0]):
            same_population = False
            # For neurons of the same types, considering the stimulation among themselves
            if pre == post:
                same_population = True
            # Forming the connection index between different types of neurons
            con = create_synapses(N[pre], N[post], p[pre][post], same_population)

            # Go through every column -> aka go through all the synpases
            for i in range(con.shape[1]):
                # identifying the corresponding pre/post neurons of the synpase
                pre_neuron = con[0][i]
                post_neuron = con[1][i]
                
                # region: How much should we skip for the target population
                step_pre = np.sum(N[:pre])
                step_post = np.sum(N[:post])

                # Adding the connectivity between the two neurons in the final matrix
                W_rec[step_pre + pre_neuron][step_post + post_neuron] = normal_distr_weights(w_initial[pre][post],
                                                                                             w_noise)
    # Why tranposed? 
    return (W_rec.T)

def calculate_selectivity_test(activity_popu):
    """
    Calculate mean and std of selectivity.

    """

    os_mean_data = []  # orientation selectivity
    ds_mean_data = []  # directions selectivity
    os_paper_mean_data = []  # directions selectivity calculated as in paper

    for population in range(len(activity_popu)):
        preferred_orientation = np.argmax(activity_popu[population], axis=0)
        #print('Popu:',population)
        #print('0',np.count_nonzero(preferred_orientation == 0))
        #print('1', np.count_nonzero(preferred_orientation == 1))
        #print('2', np.count_nonzero(preferred_orientation == 2))
        #print('3', np.count_nonzero(preferred_orientation == 3))

        os, ds, os_paper = [], [], []

        for neuron in range(activity_popu[population].shape[1]):
            s_max_index = preferred_orientation[neuron]

            # activity of preferred stimulus
            s_pref = activity_popu[population][s_max_index][neuron]

            # activity of preferred orientation in both directions
            s_pref_orient = np.mean([activity_popu[population][s_max_index][neuron],
                                    activity_popu[population][(s_max_index + 2) % 4][neuron]])

            # activity of orthogonal stimulus
            s_orth = np.mean([activity_popu[population][(s_max_index + 1) % 4][neuron],
                             activity_popu[population][(s_max_index + 3) % 4][neuron]])

            # activity of opposite stimulus
            s_oppo = activity_popu[population][(s_max_index + 2) % 4][neuron]

            os.append(np.abs(s_pref_orient - s_orth) / (s_pref_orient + s_orth))
            ds.append(np.abs(s_pref - s_oppo) / (s_pref + s_oppo))
            os_paper.append(np.abs(s_pref - s_orth) / (s_pref + s_orth))

        os_mean_data.append(np.mean(os))
        ds_mean_data.append(np.mean(ds))
        os_paper_mean_data.append(np.mean(os_paper))

    return (os_mean_data,ds_mean_data,os_paper_mean_data)

def calculate_selectivity(activity_popu):
    """
    Calculate mean and std of selectivity.
    No need to be used in my project.
    """

    os_mean_data = []  # orientation selectivity
    os_std_data = []
    ds_mean_data = []  # directions selectivity
    ds_std_data = []
    os_paper_mean_data = []  # directions selectivity calculated as in paper
    os_paper_std_data = []

    for population in range(4):
        preferred_orientation = np.argmax(activity_popu[population], axis=0)

        os, ds, os_paper = [], [], []
        preferred_orientation_freq = [0, 0, 0, 0]

        for neuron in range(activity_popu[population].shape[1]):
            s_max_index = preferred_orientation[neuron]
            preferred_orientation_freq[s_max_index] += 1

            # activity of preferred stimulus
            s_pref = activity_popu[population][s_max_index][neuron]

            # activity of preferred orientation in both directions
            s_pref_orient = np.mean([activity_popu[population][s_max_index][neuron],
                                    activity_popu[population][(s_max_index + 2) % 4][neuron]])

            # activity of orthogonal stimulus
            s_orth = np.mean([activity_popu[population][(s_max_index + 1) % 4][neuron],
                             activity_popu[population][(s_max_index + 3) % 4][neuron]])

            # activity of opposite stimulus
            s_oppo = activity_popu[population][(s_max_index + 2) % 4][neuron]

            os.append(np.abs(s_pref_orient - s_orth) / (s_pref_orient + s_orth))
            ds.append(np.abs(s_pref - s_oppo) / (s_pref + s_oppo))
            os_paper.append(np.abs(s_pref - s_orth) / (s_pref + s_orth))

        os_mean_data.append(np.mean(os))
        os_std_data.append(np.std(os))
        ds_mean_data.append(np.mean(ds))
        ds_std_data.append(np.std(ds))
        os_paper_mean_data.append(np.mean(os_paper))
        os_paper_std_data.append(np.std(os_paper))

    return (os_mean_data, os_std_data,ds_mean_data,ds_std_data,os_paper_mean_data,os_paper_std_data)

def plot_activity(activity, N, title,sim):

    '''
    activity: 3d matrix with infomraiton on the strength of the synpase? 
    '''
    if len(activity) == 0:
        return(0)
    # Extract the connectivity data for each cell population? 
    
    activity_cs = activity[:, :, :N[0]]
    activity_cc = activity[:, :, sum(N[:1]):sum(N[:2])]
    activity_pv = activity[:, :, sum(N[:2]):sum(N[:3])]
    activity_sst = activity[:, :, sum(N[:3]):sum(N)]

    for g in range(activity_cs.shape[0]): # degrees
        fig,axs = plt.subplots()
        for i in range(activity_cs.shape[2]):
            plt.plot(range(activity_cs.shape[1]),activity_cs[g,:,i],c='grey',alpha=0.5)
        plt.title('CS')
        title_save = title+ '/' + str(sim) + str(g)+ '_CS.png'
        fig.savefig(title_save)

        fig, axs = plt.subplots()
        for i in range(activity_cc.shape[2]):
            plt.plot(range(activity_cc.shape[1]), activity_cc[g,:,i], c='grey', alpha=0.5)
        plt.title('CC')
        title_save = title + '/' + str(sim) + str(g) + '_CC.png'
        fig.savefig(title_save)

        fig, axs = plt.subplots()
        for i in range(activity_pv.shape[2]):
            plt.plot(range(activity_pv.shape[1]), activity_pv[g,:,i], c='grey', alpha=0.5)
        plt.title('PV')
        title_save = title + '/' + str(sim) + str(g) + '_PV.png'
        fig.savefig(title_save)

        fig, axs = plt.subplots()
        for i in range(activity_sst.shape[2]):
            plt.plot(range(activity_sst.shape[1]), activity_sst[g,:,i], c='grey', alpha=0.5)
        plt.title('SST')
        title_save = title + '/' + str(sim) + str(g) + '_SST.png'
        fig.savefig(title_save)

