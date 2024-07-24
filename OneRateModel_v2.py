import pytensor
import pytensor.tensor as pt
import pymc as pm
import numpy as np
import scipy.io as spio
from os.path import dirname, realpath, join as pjoin

def main():
    # Get data
    dir_path = dirname(realpath(__file__))
    mat_fname = pjoin(dir_path, 'dataForBayesianAnalysisEEG-VM.mat')
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

        A1mu = pm.Normal('A1mu', mu=4, sigma=1)
        A1std = pm.Gamma('A1std', mu=0.5, sigma=0.25)
        B1mu = pm.Normal('B1mu', mu=-2, sigma=1)
        B1std = pm.Gamma('B1std', mu=0.5, sigma=0.25)
        ratio1mu = pm.Normal('ratio1mu', mu=0, sigma=1)
        ratio1std = pm.Gamma('ratio1std', mu=0.5, sigma=0.25)
        totalvarmu = pm.Gamma('totalvarmu', mu=16, sigma=4)
        totalvarstd = pm.Gamma('totalvarstd', mu=4, sigma=1)

        A1snorm = pm.Normal('A1snorm', mu=0, sigma=1, dims='subjects')
        A1 = A1mu+A1std*A1snorm
        B1snorm = pm.Normal('B1snorm', mu=0, sigma=1, dims='subjects')
        B1 = B1mu+B1std*B1snorm
        ratiovar1snorm = pm.Normal('ratiovar1snorm', mu=0, sigma=1, dims='subjects')
        ratiovar1 = ratio1mu+ratio1std*ratiovar1snorm
        ratiovar = pm.math.invlogit(ratiovar1)

        A = pm.Deterministic('A', pm.math.invlogit(A1))
        B = pm.Deterministic('B', pm.math.invlogit(B1))
        totalvar = pm.Gamma('totalvar', mu=totalvarmu, sigma=totalvarstd, dims='subjects')
        var_eta = totalvar * ratiovar
        var_epsilon = totalvar * (1-ratiovar)
        sigma_eta = pm.Deterministic('sigma_eta',pm.math.sqrt(var_eta))
        sigma_epsilon = pm.Deterministic('sigma_epsilon',pm.math.sqrt(var_epsilon))

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
        idata = pm.sample(cores=1,chains=1,draws=5000,tune=5000,init='adapt_diag')
        idata.to_netcdf(posterior_fname)

if __name__ == '__main__':
    main()