from os.path import abspath, sep, pardir
import sys
sys.path.append(abspath('') + sep + pardir + sep )
import numpy as np
import time
import tools as snt
import integration_methods as im

class SimpleNetwork:
    def __init__(self,
                W_rec,
                W_project,
                nonlinearity_rule,
                integrator='forward_euler',
                delta_t=0.01, 
                tau=5.,
                tau_learn=1000,
                tau_threshold=1000,
                Ttau=6,
                update_function='version_normal',
                learning_rule='none',
                gamma=1):
        self.W_rec =W_rec
        self.W_input=W_project
        self.W_structure=np.ones((W_rec.shape[0], W_project.shape[-1]))
        self.delta_t = delta_t
        self.tau=tau
        self.tau_learn=tau_learn
        self.tau_threshold=tau_threshold
        self.tsteps=int((Ttau*np.max(tau))/delta_t)
        self.number_steps_before_learning = 5000
        self.number_timepoints_plasticity = int(-1*(self.tau_threshold/self.delta_t)*np.log(0.1))
        self.update_function=update_function
        self.learning_rule=learning_rule
        self.gamma=gamma
        
        self.nonlinearity_rule = nonlinearity_rule
        self.integrator = integrator    
        self.inputs = None        
        self.start_activity = 0.
        self._init_nonlinearity()
        self._init_update_function()
        self._init_learningrule()
        self._init_integrator()
        
    def _init_nonlinearity(self):
        if self.nonlinearity_rule=='supralinear':
            self.np_nonlinearity = snt.supralinear(self.gamma)
        elif self.nonlinearity_rule == 'sigmoid':
            self.t_nonlinearity = snt.nl_sigmoid
        elif self.nonlinearity_rule == 'tanh':
            self.t_nonlinearity = snt.nl_tanh
            
    def _init_update_function(self):
        if self.update_function=='version_normal':
            self.update_act = im.update_network
            
    def _init_learningrule(self):
        if self.learning_rule=='none':
            self.learningrule = im.nonlearning_weights  
        if self.learning_rule=='BCM_rule_test':
            self.learningrule = BCM_rule_test
        if self.learning_rule=='BCM':
            self.learningrule = im.BCM_rule_sliding_th
            
    def _init_integrator(self):        
        if self.integrator == 'runge_kutta':
            self.integrator_function = im.runge_kutta_explicit    
        elif self.integrator == 'forward_euler':
            self.integrator_function = im.forward_euler
        else:
            raise Exception("Unknown integrator ({0})".format(self.integrator))
    
    def _check_input(self, inputs):
        '''
        Make sure the input is of the same type and structure.
        '''
        Ntotal = self.tsteps
        assert inputs.shape[-1]==self.W_input.shape[-1]
        if len(inputs.shape)==1:            
            inputs_time=np.tile(inputs, (Ntotal,1))
        if len(inputs.shape)==2:
            assert inputs.shape[0]==Ntotal
            inputs_time=inputs
        return inputs_time
            
    def run(self, inputs, start_activity):
        Ntotal = self.tsteps
        all_act=[]
        all_act.append(start_activity)
        
        inputs_time = self._check_input(inputs)
        for step in range(Ntotal):
            # For each update, first update the act such that the neuron activation level get updated
            new_act=self.integrator_function(self.update_act,  #intergation method
                              all_act[-1],  #general parameters     
                              delta_t=self.delta_t,
                              tau=self.tau, w_rec=self.W_rec , w_input=self.W_input, #kwargs
                              Input=inputs_time[step],            
                              nonlinearity=self.np_nonlinearity, )
            all_act.append(new_act) # append new activity to use for learning
            
        return all_act
            
        
        
        
    