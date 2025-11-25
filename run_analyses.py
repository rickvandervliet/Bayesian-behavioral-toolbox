from scipy.io import loadmat
import pymc as pm
import arviz as az
import numpy as np
import pickle
from behavioral_models.motor_adaptation import run_motor_adaptation_model
import os.path

def generate_data(perturbation, vision, priors):
    num_subjects = perturbation.shape[0]
    num_trials = perturbation.shape[1]
    
    if priors["rate"]==1:
        # Model parameters
        A_gen = np.random.normal(priors["A_logit_mu_mu"],priors["A_logit_sd_mu"],num_subjects)
        A_gen = 1/(1+(np.exp((-A_gen))))
        B_gen = np.random.normal(priors["B_logit_mu_mu"],priors["B_logit_sd_mu"],num_subjects)
        B_gen = 1/(1+(np.exp((-B_gen))))
        var_total_mu = priors["var_total_mu_mu"]
        var_total_var = np.pow(priors["var_total_sd_mu"],2)
        scale = var_total_var/var_total_mu
        shape = var_total_mu/scale
        var_total = np.random.gamma(shape, scale, num_subjects)
        ratio = np.random.normal(priors["ratio_logit_mu_mu"],priors["ratio_logit_sd_mu"],num_subjects)
        ratio = 1/(1+(np.exp((-ratio))))
        sigma_eta_gen = np.sqrt(var_total * ratio)
        sigma_epsilon_gen = np.sqrt(var_total * (1-ratio))
        
        # initialize
        x = np.zeros((num_subjects, num_trials+1))
        y_obs = np.zeros((num_subjects, num_trials))
        eta_gen = np.random.normal(0, sigma_eta_gen.reshape(-1,1), [num_subjects, num_trials+1])
        epsilon_gen = np.random.normal(0, sigma_epsilon_gen.reshape(-1,1), [num_subjects,num_trials])
        gen_params = {'A_gen': A_gen, 'B_gen': B_gen, 'sigma_eta_gen': sigma_eta_gen, 'sigma_epsilon_gen': sigma_epsilon_gen}

        # step 0
        x[0,:] = eta_gen[0,:]

        # update
        for s in range(num_subjects):
            for t in range(num_trials):
                y_obs[s,t] = x[s,t] + perturbation[s,t] + epsilon_gen[s,t]
                x[s,t+1] = A_gen[s] * x[s,t] - vision[s,t] * B_gen[s] * y_obs[s,t] + eta_gen[s,t+1]
                
    elif priors["rate"]==2:
        # Model parameters
        imAf_logit = np.random.normal(priors["imAf_logit_mu_mu"], priors["imAf_logit_sd_mu"], num_subjects)  
        imAf = 1/(1+(np.exp((-imAf_logit))))
        Af_gen = 1-imAf
        imAs_logit = np.random.normal(priors["imAs_logit_mu_mu"], priors["imAs_logit_sd_mu"], num_subjects)
        imAs = 1/(1+(np.exp((-imAs_logit))))
        As_gen = 1 - imAf * imAs
        Bf_logit = np.random.normal(priors["Bf_logit_mu_mu"], priors["Bf_logit_sd_mu"], num_subjects)
        Bf_gen = 1/(1+(np.exp((-Bf_logit))))
        Bs1_logit = np.random.normal(priors["Bs1_logit_mu_mu"], priors["Bs1_logit_sd_mu"], num_subjects)
        Bs_gen = 1/(1+(np.exp(-(Bf_logit+Bs1_logit))))
        var_total_mu = priors["var_total_mu_mu"]
        var_total_var = np.pow(priors["var_total_sd_mu"],2)
        scale = var_total_var/var_total_mu
        shape = var_total_mu/scale
        var_total = np.random.gamma(shape, scale, num_subjects)
        ratio = np.random.normal(priors["ratio_logit_mu_mu"],priors["ratio_logit_sd_mu"],num_subjects)
        ratio = 1/(1+(np.exp((-ratio))))
        sigma_eta_gen = np.sqrt(var_total * ratio / 2)
        sigma_epsilon_gen = np.sqrt(var_total * (1-ratio))
        
        # initialize
        xf = np.zeros((num_subjects, num_trials+1))
        xs = np.zeros((num_subjects, num_trials+1))
        y_obs = np.zeros((num_subjects, num_trials))
        etaf_gen = np.random.normal(0, sigma_eta_gen.reshape(-1,1), [num_subjects, num_trials+1])
        etas_gen = np.random.normal(0, sigma_eta_gen.reshape(-1,1), [num_subjects, num_trials+1])
        epsilon_gen = np.random.normal(0, sigma_epsilon_gen.reshape(-1,1), [num_subjects,num_trials])
        gen_params = {'Af_gen': Af_gen, 'As_gen': As_gen, 'Bf_gen': Bf_gen, 'Bs_gen': Bs_gen,
                           'sigma_eta_gen': sigma_eta_gen, 'sigma_epsilon_gen': sigma_epsilon_gen}

        # step 0
        xf[0,:] = etaf_gen[0,:]
        xs[0,:] = etaf_gen[0,:]

        # update
        for s in range(num_subjects):
            for t in range(num_trials):
                y_obs[s,t] = xf[s,t] + xs[s,t] + perturbation[s,t] + epsilon_gen[s,t]
                xf[s,t+1] = Af_gen[s] * xf[s,t] - vision[s,t] * Bf_gen[s] * y_obs[s,t] + etaf_gen[s,t+1]
                xs[s,t+1] = As_gen[s] * xs[s,t] - vision[s,t] * Bs_gen[s] * y_obs[s,t] + etas_gen[s,t+1]
                    
    return y_obs,gen_params

def main():
    # Load data from Jonker - Neuroimage - 2021
    data_mat = loadmat('behavioral_data/dataForBayesianAnalysisEEG-VM.mat', simplify_cells=True)
    aimingError = np.transpose(data_mat['aimingError'])
    rotation = np.transpose(data_mat['rotation'])
    showcursor = np.transpose(data_mat['showCursor'])

    Y = aimingError + rotation
    Y[np.absolute(Y)>30] = np.nan
    V = showcursor
    P = rotation

    # Run one rate hierarchical model
    fname = 'output/jonker-neuroimage-2021/motor-adaptation-one-rate.nc'
    if not os.path.isfile(fname):
        idata,mod = run_motor_adaptation_model(Y, V, P, two_rate = False, hierarchical=True)
        pm.sample_posterior_predictive(idata,mod,extend_inferencedata=True)
        idata.to_netcdf(fname)
    else:
        print('Using existing fit for the one rate model.')

    # Run two rate hierarchical model
    fname = 'output/jonker-neuroimage-2021/motor-adaptation-two-rate.nc'
    if not os.path.isfile(fname):
        idata,mod = run_motor_adaptation_model(Y, V, P, two_rate = True, hierarchical=True)
        pm.sample_posterior_predictive(idata,mod,extend_inferencedata=True)
        idata.to_netcdf(fname)
    else:
        print('Using existing fit for the two rate model.')        

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
            'ratio_logit_sd_mu':p_t_r.ratio_logit_sd.mean().to_numpy(),'ratio_logit_sd_sd':p_t_r.ratio_logit_sd.std().to_numpy()
            }

    # Generate data for the one and two rate adaptation models
    num_subjects = 100
    
    perturbation = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                             0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4,
                             4, 4, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8, 6,
                             6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2,
                             2, 2, 0, 0, 0, 0, 0, 0, -2, -2, -2, -2, -2, -2, -4,
                             -4, -4, -4, -4, -4, -6, -6, -6, -6, -6, -6, -8, -8,
                             -8, -8, -8, -8, -6, -6, -6, -6, -6, -6, -4, -4, -4,
                             -4, -4, -4, -2, -2, -2, -2, -2, -2])
    perturbation = np.vstack([perturbation] * num_subjects)
    
    vision = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
                        1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1,
                        1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
                        1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0,
                        1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 
                        1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
                        0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1])
    vision = np.vstack([vision] * num_subjects)
    
    y_obs_one_rate,gen_params_one_rate = generate_data(perturbation, vision, p_o_r)
    y_obs_two_rate,gen_params_two_rate = generate_data(perturbation, vision, p_t_r)
    with open('output/simulations/gen_params_one_rate.pickle', 'wb') as handle:
        pickle.dump(gen_params_one_rate, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_obs_one_rate, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('output/simulations/gen_params_two_rate.pickle', 'wb') as handle:        
        pickle.dump(gen_params_two_rate, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_obs_two_rate, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #Estimate parameters with Bayesian or EM implementations of one and two rate models
    idata,mod = run_motor_adaptation_model(y_obs_one_rate, vision, perturbation, two_rate=False, hierarchical=False)
    pm.sample_posterior_predictive(idata,mod,extend_inferencedata=True)
    idata.to_netcdf('output/simulations/data-one-rate-model-one-rate-non-hierarchical.nc')
    
    idata,mod = run_motor_adaptation_model(y_obs_one_rate, vision, perturbation, two_rate=False, hierarchical=True)
    pm.sample_posterior_predictive(idata,mod,extend_inferencedata=True)
    idata.to_netcdf('output/simulations/data-one-rate-model-one-rate-hierarchical.nc')
    
    idata,mod = run_motor_adaptation_model(y_obs_one_rate, vision, perturbation, two_rate=True, hierarchical=True)
    pm.sample_posterior_predictive(idata,mod,extend_inferencedata=True)
    idata.to_netcdf('output/simulations/data-one-rate-model-two-rate-hierarchical.nc')
    
    idata,mod = run_motor_adaptation_model(y_obs_two_rate, vision, perturbation, two_rate=False, hierarchical=True)
    pm.sample_posterior_predictive(idata,mod,extend_inferencedata=True)
    idata.to_netcdf('output/simulations/data-two-rate-model-one-rate-hierarchical.nc')
    
    idata,mod = run_motor_adaptation_model(y_obs_two_rate, vision, perturbation, two_rate=True, hierarchical=False)
    pm.sample_posterior_predictive(idata,mod,extend_inferencedata=True)
    idata.to_netcdf('output/simulations/data-two-rate-model-two-rate-non-hierarchical.nc')
    
    idata,mod = run_motor_adaptation_model(y_obs_two_rate, vision, perturbation, two_rate=True, hierarchical=True)
    pm.sample_posterior_predictive(idata,mod,extend_inferencedata=True)
    idata.to_netcdf('output/simulations/data-two-rate-model-two-rate-hierarchical.nc')
    
if __name__ == '__main__':
    main()