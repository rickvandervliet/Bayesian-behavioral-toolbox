import numpy as np
from pymc_extras.statespace.core.statespace import PyMCStateSpace
import pytensor.tensor as pt
import pymc as pm
import platform

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

import numpy as np
from pymc_extras.statespace.core.statespace import PyMCStateSpace
import pytensor.tensor as pt

### Two rate model class
class motor_adaptation_two_rate(PyMCStateSpace):
    def __init__(self, name, perturbation, vision, Af, As, Bf, Bs, var_eta, var_epsilon, num_trials):
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
        self.var_eta = var_eta
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
        self.register_variable("var_eta",self.var_eta) #sigma_eta
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
            pt.stack([self.var_eta, 0.0, 0]),
            pt.stack([0, self.var_eta, 0]),
            pt.stack([0.0, 0.0, self.var_epsilon])
        ])

        state_intercept = pt.zeros((self.num_trials, self.k_states))
        state_intercept = state_intercept[:, 2].set(self.perturbation)
        self.ssm["state_intercept"] = state_intercept

        self.ssm["design", :, :] = np.array([[0, 0, 1]])

    @property
    def param_names(self):
            return ["x0", "P0", "Af", "As", "Bf", "Bs", "var_eta", "var_epsilon"]
     
    @property
    def state_names(self):
        return ["Xf", "Xs", "Yfs"]
    
    @property
    def observed_states(self):
        return ["handerror"]
    
    @property
    def shock_names(self):
        return ["eta", "eta", "epsilon"]

def build_motor_adaptation_one_rate_model(subject_id, Y, P, V, A, B, var_eta, var_epsilon, num_trials):
    tsm = motor_adaptation_one_rate(name=f"S{subject_id}", perturbation=P, vision=V, A=A, B=B, var_eta=var_eta, var_epsilon=var_epsilon, num_trials=num_trials)
    tsm.build_statespace_graph(data=Y)

def build_motor_adaptation_two_rate_model(subject_id, Y, P, V, Af, As, Bf, Bs, var_etaf, var_etas, var_epsilon, num_trials):
    tsm = motor_adaptation_two_rate(name=f"S{subject_id}", perturbation=P, vision=V, Af=Af, As=As, Bf=Bf, Bs=Bs, var_etaf=var_etaf, var_etas=var_etas, var_epsilon=var_epsilon, num_trials=num_trials)
    tsm.build_statespace_graph(data=Y)

def run_motor_adaptation_model(Y, V, P, two_rate, hierarchical, **kwargs):
    num_subjects = Y.shape[0]
    num_trials = Y.shape[1]+1
    
    Y = np.concatenate((np.zeros((num_subjects,1)), Y), axis=1)
    V = np.concatenate((V, np.zeros((num_subjects,1))), axis=1)
    P = np.concatenate((P, np.ones((num_subjects,1))), axis=1)
    
    coords = {'state': ['x', 'y'],
    'state_aux': ['x', 'y'],
    'observed_state': ['y_est'],
    'observed_state_aux': ['y_est'],
    'shock': ['var_eta', 'var_epsilon'],
    'shock_aux': ['var_eta', 'var_epsilon'],
    'subjects': np.arange(num_subjects)}
    
    priors = kwargs.get('priors', None)

    with pm.Model(coords=coords) as mod:
        if two_rate:
            if priors is None:
                priors = {'imAf_logit_mu_mu':-0.4,'imAf_logit_mu_sd':1,
                'imAf_logit_sd_mu':1,'imAf_logit_sd_sd':0.5,
                'imAs_logit_mu_mu':0,'imAs_logit_mu_sd':1,
                'imAs_logit_sd_mu':1,'imAs_logit_sd_sd':0.5,
                'Bf_logit_mu_mu':-1.5,'Bf_logit_mu_sd':1,
                'Bf_logit_sd_mu':1,'Bf_logit_sd_sd':0.5,
                'Bs1_logit_mu_mu':-1,'Bs1_logit_mu_sd':1,
                'Bs1_logit_sd_mu':1,'Bs1_logit_sd_sd':0.5,
                'var_total_mu_mu':7,'var_total_mu_sd':1,
                'var_total_sd_mu':1,'var_total_sd_sd':0.5,
                'ratio_logit_mu_mu':-3,'ratio_logit_mu_sd':1,
                'ratio_logit_sd_mu':1,'ratio_logit_sd_sd':0.5,
                }
            if hierarchical:
                # Shared hyperparameters
                imAf_logit_mu = pm.Normal("imAf_logit_mu", mu=priors['imAf_logit_mu_mu'], sigma=priors['imAf_logit_mu_sd'])
                imAf_logit_sd = pm.Gamma("imAf_logit_sd", mu=priors['imAf_logit_sd_mu'], sigma=priors['imAf_logit_sd_sd'])
                imAs_logit_mu = pm.Normal("imAs_logit_mu", mu=priors['imAs_logit_mu_mu'], sigma=priors['imAs_logit_mu_sd'])
                imAs_logit_sd = pm.Gamma("imAs_logit_sd", mu=priors['imAs_logit_sd_mu'], sigma=priors['imAs_logit_sd_sd'])
                Bf_logit_mu = pm.Normal("Bf_logit_mu", mu=priors['Bf_logit_mu_mu'], sigma=priors['Bf_logit_mu_sd'])
                Bf_logit_sd = pm.Gamma("Bf_logit_sd", mu=priors['Bf_logit_sd_mu'], sigma=priors['Bf_logit_sd_sd'])
                Bs1_logit_mu = pm.Normal("Bs1_logit_mu", mu=priors['Bs1_logit_mu_mu'], sigma=priors['Bs1_logit_mu_sd'])
                Bs1_logit_sd = pm.Gamma("Bs1_logit_sd", mu=priors['Bs1_logit_sd_mu'], sigma=priors['Bs1_logit_sd_sd'])
                var_total_mu = pm.Gamma("var_total_mu", mu=priors['var_total_mu_mu'], sigma=priors['var_total_mu_sd'])
                var_total_sd = pm.Gamma("var_total_sd", mu=priors['var_total_sd_mu'], sigma=priors['var_total_sd_sd'])
                ratio_logit_mu = pm.Normal("ratio_logit_mu", mu=priors['ratio_logit_mu_mu'], sigma=priors['ratio_logit_mu_sd'])
                ratio_logit_sd = pm.Gamma("ratio_logit_sd", mu=priors['ratio_logit_sd_mu'], sigma=priors['ratio_logit_sd_sd'])

                # Subject-level parameters (with dims)
                imAf_logit = pm.Normal("imAf_logit", mu=imAf_logit_mu, sigma=imAf_logit_sd, dims=["subjects"])  
                imAf = pm.Deterministic("imAf", pm.math.sigmoid(imAf_logit), dims=["subjects"]) # Centered around sigmoid(-0.4) ≈ 0.4 = imA
                Af = pm.Deterministic("Af", 1.0 - imAf, dims=["subjects"])                      # Af = 0.6
                imAs_logit = pm.Normal("imAs_logit", mu=imAs_logit_mu, sigma=imAs_logit_sd, dims=["subjects"])
                imAs = pm.Deterministic("imAs", pm.math.sigmoid(imAs_logit), dims=["subjects"]) # Centered around sigmoid(0) ≈ 0.5 = imAs
                As = pm.Deterministic("As", 1.0 - imAf * imAs, dims=["subjects"])  #  As > Af
                Bf_logit = pm.Normal("Bf_logit", mu=Bf_logit_mu, sigma=Bf_logit_sd, dims=["subjects"])  # -2 < Bf_logit < -1
                Bf = pm.Deterministic("Bf", pm.math.sigmoid(Bf_logit), dims=["subjects"]) # 0.12 < Bf < 0.27   
                Bs1_logit = pm.Normal("Bs1_logit", mu=Bs1_logit_mu, sigma=Bs1_logit_sd, dims=["subjects"])  # -2 < Bs1_logit < 0
                Bs_logit = pm.Deterministic("Bs_logit", Bf_logit + Bs1_logit, dims=["subjects"])                  # Bs_logit < Bf_logit
                Bs = pm.Deterministic("Bs", pm.math.sigmoid(Bs_logit), dims=["subjects"]) # Bs < Bf
                var_total = pm.Gamma("var_total", mu=var_total_mu, sigma=var_total_sd, dims=["subjects"])
                ratio_logit = pm.Normal("ratio_logit", mu=ratio_logit_mu, sigma=ratio_logit_sd, dims=["subjects"])  
                ratio = pm.Deterministic("ratio", pm.math.sigmoid(ratio_logit), dims=["subjects"])
                var_eta = pm.Deterministic("var_eta", var_total * ratio / 2, dims=["subjects"])
                sigma_eta = pm.Deterministic("sigma_eta", pm.math.sqrt(var_eta), dims=["subjects"])
                var_epsilon = pm.Deterministic("var_epsilon", var_total * (1 - ratio), dims=["subjects"])
                sigma_epsilon = pm.Deterministic("sigma_epsilon", pm.math.sqrt(var_epsilon), dims=["subjects"])
            else:
                # Subject-level parameters (with dims)
                imAf_logit = pm.Normal("imAf_logit", mu=priors['imAf_logit_mu_mu'], sigma=priors['imAf_logit_sd_mu'], dims=["subjects"])  
                imAf = pm.Deterministic("imAf", pm.math.sigmoid(imAf_logit), dims=["subjects"]) # Centered around sigmoid(-0.4) ≈ 0.4 = imA
                Af = pm.Deterministic("Af", 1.0 - imAf, dims=["subjects"])                      # Af = 0.6
                imAs_logit = pm.Normal("imAs_logit", mu=priors['imAs_logit_mu_mu'], sigma=priors['imAs_logit_sd_mu'], dims=["subjects"])
                imAs = pm.Deterministic("imAs", pm.math.sigmoid(imAs_logit), dims=["subjects"]) # Centered around sigmoid(0) ≈ 0.5 = imAs
                As = pm.Deterministic("As", 1.0 - imAf * imAs, dims=["subjects"])  #  As > Af
                Bf_logit = pm.Normal("Bf_logit", mu=priors['Bf_logit_mu_mu'], sigma=priors['Bf_logit_sd_mu'], dims=["subjects"])  # -2 < Bf_logit < -1
                Bf = pm.Deterministic("Bf", pm.math.sigmoid(Bf_logit), dims=["subjects"]) # 0.12 < Bf < 0.27   
                Bs1_logit = pm.Normal("Bs1_logit", mu=priors['Bs1_logit_mu_mu'], sigma=priors['Bs1_logit_sd_mu'], dims=["subjects"])  # -2 < Bs1_logit < 0
                Bs_logit = pm.Deterministic("Bs_logit", Bf_logit + Bs1_logit, dims=["subjects"])                  # Bs_logit < Bf_logit
                Bs = pm.Deterministic("Bs", pm.math.sigmoid(Bs_logit), dims=["subjects"]) # Bs < Bf
                var_total = pm.Gamma("var_total", mu=priors['var_total_mu_mu'], sigma=priors['var_total_sd_mu'], dims=["subjects"])
                ratio_logit = pm.Normal("ratio_logit", mu=priors['ratio_logit_mu_mu'], sigma=priors['ratio_logit_sd_mu'], dims=["subjects"])  
                ratio = pm.Deterministic("ratio", pm.math.sigmoid(ratio_logit), dims=["subjects"])
                var_eta = pm.Deterministic("var_eta", var_total * ratio / 2, dims=["subjects"])
                sigma_eta = pm.Deterministic("sigma_eta", pm.math.sqrt(var_eta), dims=["subjects"])
                var_epsilon = pm.Deterministic("var_epsilon", var_total * (1 - ratio), dims=["subjects"])
                sigma_epsilon = pm.Deterministic("sigma_epsilon", pm.math.sqrt(var_epsilon), dims=["subjects"])

            # Use for loop instead of scan to avoid symbolic naming issues
            for i in range(num_subjects):
                # Create subject-specific variables with the correct prefix
                x0_i = pm.Data(f"S{i}_x0", np.zeros(3, dtype="float"))
                P0_i = pm.Data(f"S{i}_P0", np.eye(3))
                Af_i = pm.Deterministic(f"S{i}_Af", Af[i])
                As_i = pm.Deterministic(f"S{i}_As", As[i])
                Bf_i = pm.Deterministic(f"S{i}_Bf", Bf[i])
                Bs_i = pm.Deterministic(f"S{i}_Bs", Bs[i])
                var_eta_i = pm.Deterministic(f"S{i}_var_eta", var_eta[i])
                var_epsilon_i = pm.Deterministic(f"S{i}_var_epsilon", var_epsilon[i])
                
                build_motor_adaptation_two_rate_model(
                    subject_id=i,
                    Y=Y[i].reshape(-1, 1),  # Reshape to 2D: (n_timesteps, 1)
                    P=P[i],
                    V=V[i],
                    Af=Af_i,
                    As=As_i,
                    Bf=Bf_i,
                    Bs=Bs_i,
                    var_eta=var_eta_i,
                    var_epsilon=var_epsilon_i,
                    num_trials=num_trials
                )
        else:
            if priors is None:
                priors = {'A_logit_mu_mu':4,'A_logit_mu_sd':1,
                'A_logit_sd_mu':1,'A_logit_sd_sd':0.5,
                'B_logit_mu_mu':-2,'B_logit_mu_sd':1,
                'B_logit_sd_mu':1,'B_logit_sd_sd':0.5,
                'var_total_mu_mu':7,'var_total_mu_sd':1,
                'var_total_sd_mu':1,'var_total_sd_sd':0.5,
                'ratio_logit_mu_mu':-3,'ratio_logit_mu_sd':1,
                'ratio_logit_sd_mu':1,'ratio_logit_sd_sd':0.5
                }
            if hierarchical:
                # Shared hyperparameters
                A_logit_mu = pm.Normal("A_logit_mu", mu=priors['A_logit_mu_mu'], sigma=priors['A_logit_mu_sd'])
                A_logit_sd = pm.Gamma("A_logit_sd", mu=priors['A_logit_sd_mu'], sigma=priors['A_logit_sd_sd'])
                B_logit_mu = pm.Normal("B_logit_mu", mu=priors['B_logit_mu_mu'], sigma=priors['B_logit_mu_sd'])
                B_logit_sd = pm.Gamma("B_logit_sd", mu=priors['B_logit_sd_mu'], sigma=priors['B_logit_sd_sd'])
                var_total_mu = pm.Gamma("var_total_mu", mu=priors['var_total_mu_mu'], sigma=priors['var_total_mu_sd'])
                var_total_sd = pm.Gamma("var_total_sd", mu=priors['var_total_sd_mu'], sigma=priors['var_total_sd_sd'])
                ratio_logit_mu = pm.Normal("ratio_logit_mu", mu=priors['ratio_logit_mu_mu'], sigma=priors['ratio_logit_mu_sd'])
                ratio_logit_sd = pm.Gamma("ratio_logit_sd", mu=priors['ratio_logit_sd_mu'], sigma=priors['ratio_logit_sd_sd'])

                # Subject-level parameters (with dims)
                A_logit = pm.Normal("A_logit", mu=A_logit_mu, sigma=A_logit_sd, dims=["subjects"])  
                A = pm.Deterministic("A", pm.math.sigmoid(A_logit), dims=["subjects"])
                B_logit = pm.Normal("B_logit", mu=B_logit_mu, sigma=B_logit_sd, dims=["subjects"])
                B = pm.Deterministic("B", pm.math.sigmoid(B_logit), dims=["subjects"])
                var_total = pm.Gamma("var_total", mu=var_total_mu, sigma=var_total_sd, dims=["subjects"])
                ratio_logit = pm.Normal("ratio_logit", mu=ratio_logit_mu, sigma=ratio_logit_sd, dims=["subjects"])  
                ratio = pm.Deterministic("ratio", pm.math.sigmoid(ratio_logit), dims=["subjects"])
                var_eta = pm.Deterministic("var_eta", var_total * ratio, dims=["subjects"])
                sigma_eta = pm.Deterministic("sigma_eta", pm.math.sqrt(var_eta), dims=["subjects"])
                var_epsilon = pm.Deterministic("var_epsilon", var_total * (1 - ratio), dims=["subjects"])
                sigma_epsilon = pm.Deterministic("sigma_epsilon", pm.math.sqrt(var_epsilon), dims=["subjects"])
            else:
                # Subject-level parameters (with dims)
                A_logit = pm.Normal("A_logit", mu=priors['A_logit_mu_mu'], sigma=priors['A_logit_sd_mu'], dims=["subjects"])  
                A = pm.Deterministic("A", pm.math.sigmoid(A_logit), dims=["subjects"])
                B_logit = pm.Normal("B_logit", mu=priors['B_logit_mu_mu'], sigma=priors['B_logit_sd_mu'], dims=["subjects"])
                B = pm.Deterministic("B", pm.math.sigmoid(B_logit), dims=["subjects"])
                var_total = pm.Gamma("var_total", mu=priors['var_total_mu_mu'], sigma=priors['var_total_sd_mu'], dims=["subjects"])
                ratio_logit = pm.Normal("ratio_logit", mu=priors['ratio_logit_mu_mu'], sigma=priors['ratio_logit_sd_mu'], dims=["subjects"])  
                ratio = pm.Deterministic("ratio", pm.math.sigmoid(ratio_logit), dims=["subjects"])
                var_eta = pm.Deterministic("var_eta", var_total * ratio, dims=["subjects"])
                sigma_eta = pm.Deterministic("sigma_eta", pm.math.sqrt(var_eta), dims=["subjects"])
                var_epsilon = pm.Deterministic("var_epsilon", var_total * (1 - ratio), dims=["subjects"])
                sigma_epsilon = pm.Deterministic("sigma_epsilon", pm.math.sqrt(var_epsilon), dims=["subjects"])

            # Use for loop instead of scan to avoid symbolic naming issues
            for i in range(num_subjects):
                # Create subject-specific variables with the correct prefix
                x0_i = pm.Data(f"S{i}_x0", np.zeros(2, dtype="float"))
                P0_i = pm.Data(f"S{i}_P0", np.eye(2))
                A_i = pm.Deterministic(f"S{i}_A", A[i])
                B_i = pm.Deterministic(f"S{i}_B", B[i])
                var_eta_i = pm.Deterministic(f"S{i}_var_eta", var_eta[i])
                var_epsilon_i = pm.Deterministic(f"S{i}_var_epsilon", var_epsilon[i])

                build_motor_adaptation_one_rate_model(
                    subject_id=i,
                    Y=Y[i].reshape(-1, 1),  # Reshape to 2D: (n_timesteps, 1)
                    P=P[i],
                    V=V[i],
                    A=A_i,
                    B=B_i,
                    var_eta=var_eta_i,
                    var_epsilon=var_epsilon_i,
                    num_trials=num_trials
                )
        if platform.system() == 'Windows':
            idata = pm.sample(1000, tune=1000, target_accept=0.95, model=mod, nuts_sampler="nutpie", nuts_sampler_kwargs={"backend": "jax", "gradient_backend": "jax"})
        else:     
            idata = pm.sample(1000, tune=1000, target_accept=0.95, model=mod, nuts_sampler="numpyro", nuts_sampler_kwargs={"chain_method": "vectorized"})
    return idata,mod