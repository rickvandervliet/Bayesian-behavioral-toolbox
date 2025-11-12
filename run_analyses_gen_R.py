from scipy.io import loadmat
import pymc as pm
import arviz as az
import numpy as np
import pickle
import platform
from behavioral_models.motor_adaptation_one_rate import motor_adaptation_one_rate
from behavioral_models.motor_adaptation_two_rate import motor_adaptation_two_rate
import os.path

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
                priors = {'imAf_logit_mu_mu':0,'imAf_logit_mu_sd':1,
                'imAf_logit_sd_mu':2,'imAf_logit_sd_sd':1,
                'imAs_logit_mu_mu':0,'imAs_logit_mu_sd':1,
                'imAs_logit_sd_mu':2,'imAs_logit_sd_sd':1,
                'Bf_logit_mu_mu':0,'Bf_logit_mu_sd':1,
                'Bf_logit_sd_mu':2,'Bf_logit_sd_sd':1,
                'Bs1_logit_mu_mu':0,'Bs1_logit_mu_sd':1,
                'Bs1_logit_sd_mu':2,'Bs1_logit_sd_sd':1,
                'var_total_mu_mu':2,'var_total_mu_sd':1,
                'var_total_sd_mu':2,'var_total_sd_sd':1,
                'ratio_logit_mu_mu':0,'ratio_logit_mu_sd':1,
                'ratio_logit_sd_mu':2,'ratio_logit_sd_sd':1,
                'ratio2_logit_mu_mu':0,'ratio2_logit_mu_sd':1,
                'ratio2_logit_sd_mu':2,'ratio2_logit_sd_sd':1
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
                ratio2_logit_mu = pm.Normal("ratio2_logit_mu", mu=priors['ratio2_logit_mu_mu'], sigma=priors['ratio2_logit_mu_sd'])
                ratio2_logit_sd = pm.Gamma("ratio2_logit_sd", mu=priors['ratio2_logit_sd_mu'], sigma=priors['ratio2_logit_sd_sd'])

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
                ratio2_logit = pm.Normal("ratio2_logit", mu=priors['ratio2_logit_mu_mu'], sigma=priors['ratio2_logit_sd_mu'], dims=["subjects"])  
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
            if priors is None:
                priors = {'A_logit_mu_mu':0,'A_logit_mu_sd':1,
                'A_logit_sd_mu':2,'A_logit_sd_sd':1,
                'B_logit_mu_mu':0,'B_logit_mu_sd':1,
                'B_logit_sd_mu':2,'B_logit_sd_sd':1,
                'var_total_mu_mu':2,'var_total_mu_sd':1,
                'var_total_sd_mu':2,'var_total_sd_sd':1,
                'ratio_logit_mu_mu':0,'ratio_logit_mu_sd':1,
                'ratio_logit_sd_mu':2,'ratio_logit_sd_sd':1
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

def main():
    # Load Gen R data
    # To do
    
    param_file = 'output/jonker-neuroimage-2021/motor-adaptation-one-rate.nc'
    p_o_r = az.from_netcdf(param_file)
    p_o_r = p_o_r.posterior
    p_o_r = {'rate':1,
            'A_logit_mu_mu': p_o_r.A_logit_mu.mean().to_numpy(),'A_logit_mu_sd':p_o_r.A_logit_mu.std().to_numpy(),
            'A_logit_sd_mu':p_o_r.A_logit_sd.mean().to_numpy(),'A_logit_sd_sd':p_o_r.A_logit_sd.std().to_numpy(),
            'B_logit_mu_mu':p_o_r.B_logit_mu.mean().to_numpy(),'B_logit_mu_sd':p_o_r.B_logit_mu.std().to_numpy(),
            'B_logit_sd_mu':p_o_r.B_logit_sd.mean().to_numpy(),'B_logit_sd_sd':p_o_r.B_logit_sd.std().to_numpy(),
            'var_total_mu_mu':p_o_r.var_total_mu.mean().to_numpy(),'var_total_mu_sd':p_o_r.var_total_mu.std().to_numpy(),
            'var_total_sd_mu':p_o_r.var_total_sd.mean().to_numpy(),'var_total_sd_sd':p_o_r.var_total_sd.std().to_numpy(),
            'ratio_logit_mu_mu':p_o_r.ratio_logit_mu.mean().to_numpy(),'ratio_logit_mu_sd':p_o_r.ratio_logit_mu.std().to_numpy(),
            'ratio_logit_sd_mu':p_o_r.ratio_logit_sd.mean().to_numpy(),'ratio_logit_sd_sd':p_o_r.ratio_logit_sd.std().to_numpy()
            }

    param_file = 'output/jonker-neuroimage-2021/motor-adaptation-two-rate.nc'
    p_t_r = az.from_netcdf(param_file)
    p_t_r = p_t_r.posterior
    p_t_r = {'rate':2,
            'imAf_logit_mu_mu':p_t_r.imAf_logit_mu.mean().to_numpy(),'imAf_logit_mu_sd':p_t_r.imAf_logit_mu.std().to_numpy(),
            'imAf_logit_sd_mu':p_t_r.imAf_logit_sd.mean().to_numpy(),'imAf_logit_sd_sd':p_t_r.imAf_logit_sd.std().to_numpy(),
            'imAs_logit_mu_mu':p_t_r.imAs_logit_mu.mean().to_numpy(),'imAs_logit_mu_sd':p_t_r.imAs_logit_mu.std().to_numpy(),
            'imAs_logit_sd_mu':p_t_r.imAs_logit_sd.mean().to_numpy(),'imAs_logit_sd_sd':p_t_r.imAs_logit_sd.std().to_numpy(),
            'Bf_logit_mu_mu':p_t_r.Bf_logit_mu.mean().to_numpy(),'Bf_logit_mu_sd':p_t_r.Bf_logit_mu.std().to_numpy(),
            'Bf_logit_sd_mu':p_t_r.Bf_logit_sd.mean().to_numpy(),'Bf_logit_sd_sd':p_t_r.Bf_logit_sd.std().to_numpy(),
            'Bs1_logit_mu_mu':p_t_r.Bs1_logit_mu.mean().to_numpy(),'Bs1_logit_mu_sd':p_t_r.Bs1_logit_mu.std().to_numpy(),
            'Bs1_logit_sd_mu':p_t_r.Bs1_logit_sd.mean().to_numpy(),'Bs1_logit_sd_sd':p_t_r.Bs1_logit_sd.std().to_numpy(),
            'var_total_mu_mu':p_t_r.var_total_mu.mean().to_numpy(),'var_total_mu_sd':p_t_r.var_total_mu.std().to_numpy(),
            'var_total_sd_mu':p_t_r.var_total_sd.mean().to_numpy(),'var_total_sd_sd':p_t_r.var_total_sd.std().to_numpy(),
            'ratio_logit_mu_mu':p_t_r.ratio_logit_mu.mean().to_numpy(),'ratio_logit_mu_sd':p_t_r.ratio_logit_mu.std().to_numpy(),
            'ratio_logit_sd_mu':p_t_r.ratio_logit_sd.mean().to_numpy(),'ratio_logit_sd_sd':p_t_r.ratio_logit_sd.std().to_numpy(),
            'ratio2_logit_mu_mu':p_t_r.ratio2_logit_mu.mean().to_numpy(),'ratio2_logit_mu_sd':p_t_r.ratio2_logit_mu.std().to_numpy(),
            'ratio2_logit_sd_mu':p_t_r.ratio2_logit_sd.mean().to_numpy(),'ratio2_logit_sd_sd':p_t_r.ratio2_logit_sd.std().to_numpy()
            }
    
    # Run one rate hierarchical model
    fname = 'output/Gen-R/motor-adaptation-one-rate.nc'
    if not os.path.isfile(fname):
        idata,mod = run_motor_adaptation_model(Y, V, P, two_rate = False, hierarchical=True, priors=p_o_r)
        pm.sample_posterior_predictive(idata,mod,extend_inferencedata=True)
        idata.to_netcdf(fname)

    # Run two rate hierarchical model
    fname = 'output/Gen-R/motor-adaptation-two-rate.nc'
    if not os.path.isfile(fname):
        idata,mod = run_motor_adaptation_model(Y, V, P, two_rate = True, hierarchical=True, priors=p_t_r)
        pm.sample_posterior_predictive(idata,mod,extend_inferencedata=True)
        idata.to_netcdf(fname)

if __name__ == '__main__':
    main()