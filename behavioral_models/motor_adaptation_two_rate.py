import numpy as np
from pymc_extras.statespace.core.statespace import PyMCStateSpace
import pytensor.tensor as pt

### Two rate model class
class motor_adaptation_two_rate(PyMCStateSpace):
    def __init__(self, name, perturbation, vision, Af, As, Bf, Bs, var_etaf, var_etas, var_epsilon, num_trials):
        k_states = 3  # size of the state vector x
        k_posdef = 3  # number of shocks (size of the state covariance matrix Q)
        k_endog = 1  # number of observed states

        self.perturbation = perturbation
        self.num_trials = num_trials
        self.vision = vision

        self.Af = Af
        self.As = As
        self.Bf = Bf
        self.Bs = Bs
        self.var_etaf = var_etaf
        self.var_etas = var_etas
        self.var_epsilon = var_epsilon
        
        super().__init__(k_endog=k_endog, k_states=k_states, k_posdef=k_posdef, name=name)
    
    def make_symbolic_graph(self):
        # Declare symbolic variables that represent parameters of the model
        x0 = self.make_and_register_variable("x0", shape=(3,))
        P0 = self.make_and_register_variable("P0", shape=(3, 3))
        self.register_variable("Af",self.Af) #Afast, Aslow
        self.register_variable("As",self.As)
        self.register_variable("Bf",self.Bf) #Bfast, Bslow
        self.register_variable("Bs",self.Bs)
        self.register_variable("var_etaf",self.var_etaf) #sigma_fast
        self.register_variable("var_etas",self.var_etas) #sigma_fast
        self.register_variable("var_epsilon",self.var_epsilon) #sigma_y
        
        # Next, use these symbolic variables to build the statespace matrices by assigning each parameter
        # to its correct location in the correct matrix
        self.ssm["initial_state", :] = x0
        self.ssm["initial_state_cov", :, :] = P0

        transition_vision = pt.stack([
            pt.stack([self.Af, 0, -self.Bf]),
            pt.stack([0, self.As, -self.Bs]),
            pt.stack([self.Af, self.As, -(self.Bf+self.Bs)])
        ])
        transition_no_vision = pt.stack([
            pt.stack([self.Af, 0, 0]),
            pt.stack([0, self.As, 0]),
            pt.stack([self.Af, self.As, 0])
        ])
        # Use pt.where to select the appropriate matrix for each trial
        # Reshape vision to (num_trials, 1, 1) for broadcasting
        vision_expanded = self.vision[:, None, None]
        transition = pt.where(
            pt.eq(vision_expanded, 1),
            transition_vision,
            transition_no_vision
        )
        self.ssm["transition"] = transition

        self.ssm["selection", :, :] = pt.stack([
            pt.stack([1.0, 0, 0]),
            pt.stack([0, 1, 0]),
            pt.stack([1, 1, 1])
        ])

        self.ssm["state_cov", :, :] = pt.stack([
            pt.stack([self.var_etaf, 0.0, 0]),
            pt.stack([0, self.var_etas, 0]),
            pt.stack([0.0, 0.0, self.var_epsilon])
        ])

        state_intercept = pt.zeros((self.num_trials, self.k_states))
        state_intercept = state_intercept[:, 2].set(self.perturbation)
        self.ssm["state_intercept"] = state_intercept

        self.ssm["design", :, :] = np.array([[0, 0, 1]])

    @property
    def param_names(self):
            return ["x0", "P0", "Af", "As", "Bf", "Bs", "var_etaf", "var_etas", "var_epsilon"]
     
    @property
    def state_names(self):
        return ["Xf", "Xs", "Yfs"]
    
    @property
    def observed_states(self):
        return ["handerror"]
    
    @property
    def shock_names(self):
        return ["eta_f", "eta_s", "epsilon"]