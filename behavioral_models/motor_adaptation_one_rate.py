import numpy as np
from pymc_extras.statespace.core.statespace import PyMCStateSpace
import pytensor.tensor as pt

### One rate model class
class motor_adaptation_one_rate(PyMCStateSpace):
    def __init__(self, name, perturbation, vision, A, B, var_eta, var_epsilon, num_trials):
        k_states = 2  # size of the state vector x
        k_posdef = 2  # number of shocks (size of the state covariance matrix Q)
        k_endog = 1  # number of observed states

        self.perturbation = perturbation
        self.num_trials = num_trials
        self.vision = vision

        self.A = A
        self.B = B
        self.var_eta = var_eta
        self.var_epsilon = var_epsilon

        super().__init__(k_endog=k_endog, k_states=k_states, k_posdef=k_posdef, name=name)

    def make_symbolic_graph(self):
        x0 = self.make_and_register_variable("x0", shape=(2,))
        P0 = self.make_and_register_variable("P0", shape=(2, 2))
        self.register_variable("A", self.A)
        self.register_variable("B", self.B)
        self.register_variable("var_eta", self.var_eta)
        self.register_variable("var_epsilon", self.var_epsilon)

        self.ssm["initial_state", :] = x0
        self.ssm["initial_state_cov", :, :] = P0

        # Transition matrix - time varying based on vision
        # Vision: [A-B, 0; 1, 0]
        # No vision: [A, 0; 1, 0]
        transition_vision = pt.stack([
            pt.stack([self.A, -self.B]),
            pt.stack([self.A, -self.B])
        ])
        transition_no_vision = pt.stack([
            pt.stack([self.A, 0]),
            pt.stack([self.A, 0])
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

        # Selection matrix.
        self.ssm["selection"] = pt.stack([
            pt.stack([1, 0]),
            pt.stack([1, 1])
        ])
        
        # State covariance matrix
        self.ssm["state_cov", :, :] = pt.stack([
            pt.stack([self.var_eta, 0]),
            pt.stack([0.0, self.var_epsilon])
        ])

        # State intercept matrix
        state_intercept = pt.zeros((self.num_trials, self.k_states))
        state_intercept = state_intercept[:, 1].set(self.perturbation)
        self.ssm["state_intercept"] = state_intercept

        # Design matrix
        self.ssm["design", :, :] = np.array([[0, 1]])

    @property
    def param_names(self):
        return ["x0", "P0", "A", "B", "var_eta", "var_epsilon"]
     
    @property
    def state_names(self):
        return ["x", "y"]
    
    @property
    def observed_states(self):
        return ["y_est"]
    
    @property
    def shock_names(self):
        return ["var_eta", "var_epsilon"]