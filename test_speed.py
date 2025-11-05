# -*- coding: utf-8 -*-
import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from behavioral_models.motor_adaptation_one_rate import motor_adaptation_one_rate
from behavioral_models.motor_adaptation_two_rate import motor_adaptation_two_rate
import time

# Generate data
num_trials = 110
num_subjects = 100
vision = np.ones((num_subjects, num_trials))
vision[:,5::10] = 0
perturbation = np.zeros((num_subjects, num_trials))  # initialize all to 0
perturbation[:,20:60] = 30
perturbation[:,0] = 30
perturbation[:,num_trials-1] = 30

# Model parameters
A_gen = np.random.normal(4,1,num_subjects)
A_gen = 1/(1+(np.exp((-A_gen))))
B_gen = np.random.normal(-2,1,num_subjects)
B_gen = 1/(1+(np.exp((-B_gen))))

sigma_eta_gen = np.random.gamma(2,0.5,(num_subjects,1))
sigma_epsilon_gen = np.random.gamma(4,1,(num_subjects,1))

# initialize
x = np.zeros((num_subjects, num_trials+1))
y_obs = np.zeros((num_subjects, num_trials))

# noise 
eta_gen = np.random.normal(0, sigma_eta_gen, [num_subjects, num_trials+1])
epsilon_gen = np.random.normal(0, sigma_epsilon_gen, [num_subjects,num_trials])

# step 0
x[0,:] = eta_gen[0,:]

# update
for s in range(num_subjects):
    for t in range(num_trials):
        y_obs[s,t] = x[s,t] + perturbation[s,t] + epsilon_gen[s,t]
        x[s,t+1] = A_gen[s] * x[s,t] - vision[s,t] * B_gen[s] * y_obs[s,t] + eta_gen[s,t+1]

def build_motor_adaptation_one_rate_model(subject_id, Y, P, V, A, B, var_eta, var_epsilon, num_trials):
    tsm = motor_adaptation_one_rate(name=f"S{subject_id}", perturbation=P, vision=V, A=A, B=B, var_eta=var_eta, var_epsilon=var_epsilon, num_trials=num_trials)
    tsm.build_statespace_graph(data=Y)

def build_motor_adaptation_two_rate_model(subject_id, Y, P, V, Af, As, Bf, Bs, var_etaf, var_etas, var_epsilon, num_trials):
    tsm = motor_adaptation_two_rate(name=f"S{subject_id}", perturbation=P, vision=V, Af=Af, As=As, Bf=Bf, Bs=Bs, var_etaf=var_etaf, var_etas=var_etas, var_epsilon=var_epsilon, num_trials=num_trials)
    tsm.build_statespace_graph(data=Y)

def run_motor_adaptation_model(Y, V, P, two_rate, hierarchical):
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

    with pm.Model(coords=coords) as mod:
        if two_rate:
            if hierarchical:
                # Shared hyperparameters
                imAf_logit_mu = pm.Normal("imAf_logit_mu", mu=0, sigma=1)
                imAf_logit_sd = pm.Gamma("imAf_logit_sd", mu=2, sigma=1)
                imAs_logit_mu = pm.Normal("imAs_logit_mu", mu=0, sigma=1)
                imAs_logit_sd = pm.Gamma("imAs_logit_sd", mu=2, sigma=1)
                Bf_logit_mu = pm.Normal("Bf_logit_mu", mu=0, sigma=1)
                Bf_logit_sd = pm.Gamma("Bf_logit_sd", mu=2, sigma=1)
                Bs1_logit_mu = pm.Normal("Bs1_logit_mu", mu=0, sigma=1)
                Bs1_logit_sd = pm.Gamma("Bs1_logit_sd", mu=2, sigma=1)
                var_total_mu = pm.Gamma("var_total_mu", mu=2, sigma=1)
                var_total_sd = pm.Gamma("var_total_sd", mu=2, sigma=1)
                ratio_logit_mu = pm.Normal("ratio_logit_mu", mu=0, sigma=1)
                ratio_logit_sd = pm.Gamma("ratio_logit_sd", mu=2, sigma=1)
                ratio2_logit_mu = pm.Normal("ratio2_logit_mu", mu=0, sigma=1)
                ratio2_logit_sd = pm.Gamma("ratio2_logit_sd", mu=2, sigma=1)

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
                ratio2_logit = pm.Normal("ratio2_logit", mu=ratio2_logit_mu, sigma=ratio2_logit_sd, dims=["subjects"])  
                ratio2 = pm.Deterministic("ratio2", pm.math.sigmoid(ratio2_logit), dims=["subjects"])
                var_etaf = pm.Deterministic("var_etaf", var_total * ratio * ratio2, dims=["subjects"])
                sigma_etaf = pm.Deterministic("sigma_etaf", pm.math.sqrt(var_etaf), dims=["subjects"])
                var_etas = pm.Deterministic("var_etas", var_total * ratio * (1-ratio2), dims=["subjects"])
                sigma_etas = pm.Deterministic("sigma_etas", pm.math.sqrt(var_etas), dims=["subjects"])
                var_epsilon = pm.Deterministic("var_epsilon", var_total * (1 - ratio), dims=["subjects"])
                sigma_epsilon = pm.Deterministic("sigma_epsilon", pm.math.sqrt(var_epsilon), dims=["subjects"])
            else:
                # Subject-level parameters (with dims)
                imAf_logit = pm.Normal("imAf_logit", mu=-0.4, sigma=0.15, dims=["subjects"])  
                imAf = pm.Deterministic("imAf", pm.math.sigmoid(imAf_logit), dims=["subjects"]) # Centered around sigmoid(-0.4) ≈ 0.4 = imA
                Af = pm.Deterministic("Af", 1.0 - imAf, dims=["subjects"])                      # Af = 0.6
                imAs_logit = pm.Normal("imAs_logit", mu=0, sigma=0.5, dims=["subjects"])
                imAs = pm.Deterministic("imAs", pm.math.sigmoid(imAs_logit), dims=["subjects"]) # Centered around sigmoid(0) ≈ 0.5 = imAs
                As = pm.Deterministic("As", 1.0 - imAf * imAs, dims=["subjects"])  #  As > Af
                Bf_logit = pm.Normal("Bf_logit", mu=-1.5, sigma=0.5, dims=["subjects"])  # -2 < Bf_logit < -1
                Bf = pm.Deterministic("Bf", pm.math.sigmoid(Bf_logit), dims=["subjects"]) # 0.12 < Bf < 0.27   
                Bs1_logit = pm.Normal("Bs1_logit", mu=-1.7, sigma=0.5, dims=["subjects"])  # -2 < Bs1_logit < 0
                Bs_logit = pm.Deterministic("Bs_logit", Bf_logit + Bs1_logit, dims=["subjects"])                  # Bs_logit < Bf_logit
                Bs = pm.Deterministic("Bs", pm.math.sigmoid(Bs_logit), dims=["subjects"]) # Bs < Bf
                var_total = pm.Gamma("var_total", mu=4, sigma=3, dims=["subjects"])
                ratio_logit = pm.Normal("ratio_logit", mu=0, sigma=1, dims=["subjects"])  
                ratio = pm.Deterministic("ratio", pm.math.sigmoid(ratio_logit), dims=["subjects"])
                ratio2_logit = pm.Normal("ratio2_logit", mu=0, sigma=1, dims=["subjects"])  
                ratio2 = pm.Deterministic("ratio2", pm.math.sigmoid(ratio2_logit), dims=["subjects"])
                var_etaf = pm.Deterministic("var_etaf", var_total * ratio * ratio2, dims=["subjects"])
                sigma_etaf = pm.Deterministic("sigma_etaf", pm.math.sqrt(var_etaf), dims=["subjects"])
                var_etas = pm.Deterministic("var_etas", var_total * ratio * (1-ratio2), dims=["subjects"])
                sigma_etas = pm.Deterministic("sigma_etas", pm.math.sqrt(var_etas), dims=["subjects"])
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
                var_etaf_i = pm.Deterministic(f"S{i}_var_etaf", var_etaf[i])
                var_etas_i = pm.Deterministic(f"S{i}_var_etas", var_etas[i])
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
                    var_etaf=var_etaf_i,
                    var_etas=var_etas_i,
                    var_epsilon=var_epsilon_i,
                    num_trials=num_trials
                )
        else:
            if hierarchical:
                # Shared hyperparameters
                A_logit_mu = pm.Normal("A_logit_mu", mu=0, sigma=1)
                A_logit_sd = pm.Gamma("A_logit_sd", mu=2, sigma=1)
                B_logit_mu = pm.Normal("B_logit_mu", mu=0, sigma=1)
                B_logit_sd = pm.Gamma("B_logit_sd", mu=2, sigma=1)
                var_total_mu = pm.Gamma("var_total_mu", mu=2, sigma=1)
                var_total_sd = pm.Gamma("var_total_sd", mu=2, sigma=1)
                ratio_logit_mu = pm.Normal("ratio_logit_mu", mu=0, sigma=1)
                ratio_logit_sd = pm.Gamma("ratio_logit_sd", mu=2, sigma=1)

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
                A_logit = pm.Normal("A_logit", mu=0, sigma=1, dims=["subjects"])  
                A = pm.Deterministic("A", pm.math.sigmoid(A_logit), dims=["subjects"])
                B_logit = pm.Normal("B_logit", mu=0, sigma=1, dims=["subjects"])
                B = pm.Deterministic("B", pm.math.sigmoid(B_logit), dims=["subjects"])
                var_total = pm.Gamma("var_total", mu=4, sigma=3, dims=["subjects"])
                ratio_logit = pm.Normal("ratio_logit", mu=0, sigma=1, dims=["subjects"])  
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
    idata = pm.sample(1000, tune=1000, target_accept=0.95, model=mod, nuts_sampler="nutpie", nuts_sampler_kwargs={"backend": "jax", "gradient_backend": "jax"}, progressbar=False)
    return idata,mod

def main():
    start_time = time.time()
    num_test_subjects = 100
    Y = y_obs[0:num_test_subjects,:]
    V = vision[0:num_test_subjects,:]
    P = perturbation[0:num_test_subjects,:]
    idata,mod = run_motor_adaptation_model(Y, V, P, two_rate=False, hierarchical=False)
    print("---%s seconds ----" % (time.time()-start_time))
if __name__ == '__main__':
    main()