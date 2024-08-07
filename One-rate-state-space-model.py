import pytensor
import pytensor.tensor as pt
import pymc as pm
import numpy as np
import scipy.io as spio
from os.path import dirname, realpath, join as pjoin

def main():
    # Get data
    dir_path = dirname(realpath(__file__))
    mat_fname = pjoin(dir_path, 'Behavioral-data/dataForBayesianAnalysisEEG-VM.mat')
    posterior_fname = 'posteriorDatav2.nc'
    n_steps = 900
    n_subjects = 60
    sixtydata = spio.loadmat(mat_fname)
    data_y = sixtydata['aimingError'][:n_steps,:n_subjects]
    data_p = pt.as_tensor_variable(sixtydata['rotation'][:n_steps,:n_subjects])
    data_v = pt.as_tensor_variable(sixtydata['showCursor'][:n_steps,:n_subjects])

    # PyMC Model
    coords = {'time':np.arange(n_steps),
            'subjects':np.arange(n_subjects)}

    with pm.Model(coords=coords) as model:           

        A1mu = pm.Normal('A1mu', mu=4, sigma=0.25)
        A1std = pm.Gamma('A1std', mu=0.5, sigma=0.25)
        B1mu = pm.Normal('B1mu', mu=-2, sigma=0.25)
        B1std = pm.Gamma('B1std', mu=0.5, sigma=0.25)
        etamu = pm.Gamma('etamu', mu=0.5, sigma=0.25)
        etastd = pm.Gamma('etastd', mu=0.5, sigma=0.25)
        epsilonmu = pm.Gamma('epsilonmu', mu=3, sigma=0.25)
        epsilonstd = pm.Gamma('epsilonstd', mu=0.5, sigma=0.25)

        A1 = pm.Normal('A1', mu=A1mu, sigma=A1std, dims='subjects')
        B1 = pm.Normal('B1', mu=B1mu, sigma=B1std, dims='subjects')
        A = pm.Deterministic('A', pm.math.invlogit(A1))
        B = pm.Deterministic('B', pm.math.invlogit(B1))
        sigma_eta = pm.Gamma('sigma_eta', mu=etamu, sigma=etastd, dims='subjects')
        sigma_epsilon = pm.Gamma('sigma_epsilon', mu=epsilonmu, sigma=epsilonstd, dims='subjects')

        x_init = pm.Normal('x_init', mu=0, sigma=sigma_eta)
        eta = pm.Normal('eta', mu=0, sigma=sigma_eta, dims=['time', 'subjects'])
        epsilon = pm.Normal('epsilon', mu=0, sigma=sigma_epsilon, dims=['time', 'subjects'])
        
        def grw_step(eta_t, epsilon_t, p_t, v_t, x_t, A, B):
            x_tp1 = A * x_t - v_t * B * (p_t + x_t + epsilon_t) + eta_t
            return x_tp1
        
        x, updates = pytensor.scan(fn=grw_step,
            sequences=[eta, epsilon, data_p, data_v],
            outputs_info=[{"initial": x_init}],
            non_sequences=[A,B],
            name='statespace',
            strict=True)
        
        x = pt.concatenate([[x_init], x[:-1,]], axis=0)
        y_hat = pm.Normal('y_hat',mu=x,sigma=sigma_epsilon,observed=data_y,dims=['time', 'subjects'])
        idata = pm.sample(cores=8,chains=4,draws=10000,tune=10000,init='adapt_diag')
        idata.to_netcdf(posterior_fname)

if __name__ == '__main__':
    main()