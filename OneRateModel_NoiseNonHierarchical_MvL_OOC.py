import pytensor
import pytensor.tensor as pt
import pymc as pm
import numpy as np
import scipy.io as spio
from os.path import dirname, realpath, join as pjoin
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

class TestData:
    def __init__(self, input_file, n_subjects=None, n_steps=None):

        dir_path = dirname(realpath(__file__))
        mat_fname = pjoin(dir_path, input_file)
        self.data_mat = spio.loadmat(mat_fname)

        if n_subjects is None:
            self.n_subjects = max(map(len,self.data_mat['aimingError']))
        else:
            self.n_subjects = n_subjects
             
        if n_steps is None:
            self.n_steps = len(self.data_mat['aimingError'])
        else:
            self.n_steps = n_steps

        self.y = self.data_mat['aimingError'][:self.n_steps,:self.n_subjects]
        self.p = pt.as_tensor_variable(self.data_mat['rotation'][:self.n_steps,:self.n_subjects])
        self.v = pt.as_tensor_variable(self.data_mat['showCursor'][:self.n_steps,:self.n_subjects])
        

class StateSpace:
    def __init__(self,data):
        
        self.y=data.y
        self.p=data.p
        self.v=data.v

        self.n_steps = data.n_steps
        self.n_subjects = data.n_subjects

    def OneRateNonHierarchical(self, posterior_fname = 'posteriorData-NoiseNonHierachical2.nc'):
        coords = {'time':np.arange(self.n_steps),
                'subjects':np.arange(self.n_subjects)}
        
        with pm.Model(coords=coords) as model:           
            A1mu = pm.Normal('A1mu', mu=4, sigma=0.25)
            A1std = pm.Gamma('A1std', mu=0.5, sigma=0.25)
            B1mu = pm.Normal('B1mu', mu=-2, sigma=0.25)
            B1std = pm.Gamma('B1std', mu=0.5, sigma=0.25)

            A1 = pm.Normal('A1', mu=A1mu, sigma=A1std, dims='subjects')
            B1 = pm.Normal('B1', mu=B1mu, sigma=B1std, dims='subjects')
            A = pm.Deterministic('A', pm.math.invlogit(A1), dims='subjects')
            B = pm.Deterministic('B', pm.math.invlogit(B1), dims='subjects')

            sigma_eta = pm.Gamma('sigma_eta', mu=0.5, sigma=0.1, dims='subjects')
            sigma_epsilon = pm.Gamma('sigma_epsilon', mu=4, sigma=1, dims='subjects')
            sigma_total = pm.Deterministic('sigma_total',pm.math.sqrt(sigma_eta*sigma_eta+sigma_epsilon*sigma_epsilon), dims='subjects')

            x_init = pm.Normal('x_init', mu=0, sigma=1, dims='subjects')

            eta = pm.Normal('eta', mu=0, sigma=sigma_eta, dims=['time', 'subjects'])
            epsilon = pm.Normal('epsilon', mu=0, sigma=sigma_epsilon, dims=['time', 'subjects'])
            
            def grw_step(eta_t, epsilon_t, p_t, v_t, x_t, A, B):
                x_tp1 = A * (x_t+eta_t) - v_t * B * (p_t + x_t + eta_t + epsilon_t)
                return x_tp1
            
            x, updates = pytensor.scan(fn=grw_step,
                sequences=[eta, epsilon, self.p, self.v],
                outputs_info=[{"initial": x_init}],
                non_sequences=[A,B],
                name='statespace',
                strict=True)
            
            x = pm.Deterministic('x',pt.concatenate([[x_init], x[:-1,]], axis=0), dims=['time', 'subjects'])
            y_hat = pm.Normal('y_hat',mu=x,sigma=sigma_total,observed=self.y,dims=['time', 'subjects'])
            self.idata = pm.sample(cores=4,chains=4,draws=1000,tune=1000,init='adapt_diag')
            self.idata = pm.sample_posterior_predictive(self.idata,extend_inferencedata=True)
            self.idata.to_netcdf(posterior_fname)
        
    def param_calcs(self):
        #self.idata = az.from_netcdf('posteriorData-NoiseNonHierachical2.nc') ##temp
        params = az.summary(self.idata,var_names=['A','B','sigma_eta','sigma_epsilon'])
        #params.to_excel("posteriorData-NoiseNonHierachical.xlsx")
        az.plot_ppc(self.idata, num_pp_samples=100)
        plt.show()
        


def main():
    data = TestData('Behavioral-data/dataForBayesianAnalysisEEG-VM.mat')
    ssm = StateSpace(data)
    ssm.OneRateNonHierarchical()
    ssm.param_calcs()


if __name__== '__main__':
    main()