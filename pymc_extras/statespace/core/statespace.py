# -*- coding: utf-8 -*-
import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from pymc_extras.statespace.core.statespace import PyMCStateSpace
import pytensor.tensor as pt
import pytensor

# Generate data
numTrials = 110
numSubjects = 100
vision = np.ones((numTrials, numSubjects))
vision[10::10, :] = 0
perturbation = np.zeros((numTrials, numSubjects))  # initialize all to 0
perturbation[10:55, :] = 30

# Model parameters
A = np.random.normal(4, 1, numSubjects)
A = 1 / (1 + (np.exp((-A))))
B = np.random.normal(-2, 1, numSubjects)
B = 1 / (1 + (np.exp((-B))))

sigma_eta = np.random.gamma(1, 0.5, numSubjects)
sigma_epsilon = np.random.gamma(2, 1, numSubjects)

# initialize
x = np.zeros((numTrials + 1, numSubjects))
y_obs = np.zeros((numTrials, numSubjects))

# noise
eta = np.random.normal(0, sigma_eta, [numTrials + 1, numSubjects])
epsilon = np.random.normal(0, sigma_epsilon, [numTrials, numSubjects])

# step 0
x[0, :] = eta[0, :]

# update
for s in range(numSubjects):
    for t in range(numTrials):
        y_obs[t, s] = x[t, s] + perturbation[t, s] + epsilon[t, s]
        x[t + 1, s] = A[s] * x[t, s] - vision[t, s] * B[s] * y_obs[t, s] + eta[t + 1, s]

# Plot
plt.figure(figsize=(12, 6))
plt.plot(np.mean(x, axis=1), label="State", color="tab:blue")
plt.plot(np.mean(y_obs, axis=1), label="Y_obs", color="tab:green")
plt.fill_between(
    np.arange(numTrials),
    np.mean(x[:numTrials, :], axis=1) - np.std(x[:numTrials, :], axis=1),
    np.mean(x[:numTrials, :], axis=1) + np.std(x[:numTrials, :], axis=1),
    color="tab:blue",
    alpha=0.2,
)
plt.fill_between(
    np.arange(numTrials),
    np.mean(y_obs, axis=1) - np.std(y_obs, axis=1),
    np.mean(y_obs, axis=1) + np.std(y_obs, axis=1),
    color="tab:green",
    alpha=0.2,
)
for vline in [10, 55]:
    plt.axvline(x=vline, color="gray", linestyle="--", linewidth=1)
plt.xlabel("Trial")
plt.xlim(1, 100)
plt.ylabel("Y")
plt.title("Generated data")
plt.legend()
plt.tight_layout()
plt.show()


### One rate model class
class OneRateModel(PyMCStateSpace):
    def __init__(
        self, name, perturbation, vision, A, B, var_eta, var_epsilon, numTrials
    ):
        k_states = 2  # size of the state vector x
        k_posdef = 2  # number of shocks (size of the state covariance matrix Q)
        k_endog = 1  # number of observed states

        self.perturbation = perturbation
        self.numTrials = numTrials
        self.vision = vision
        # self.vision_indices = np.argwhere(vision==1)
        # self.no_vision_indices = np.argwhere(vision==0)

        self.A = A
        self.B = B
        self.var_eta = var_eta
        self.var_epsilon = var_epsilon

        super().__init__(
            k_endog=k_endog, k_states=k_states, k_posdef=k_posdef, name=name
        )

    def make_symbolic_graph(self):
        # Declare symbolic variables that represent parameters of the model
        # To do: the prefix should be added here or in the statespace file. But if you do it here, you will still get an error.
        x0 = self.make_and_register_variable("x0", shape=(2,))
        P0 = self.make_and_register_variable("P0", shape=(2, 2))
        self.register_variable("A", self.A)
        self.register_variable("B", self.B)
        self.register_variable("var_eta", self.var_eta)
        self.register_variable("var_epsilon", self.var_epsilon)

        # I added the following function to the statespace class
        # def register_variable(self, name, variable):
        #    if name not in self.param_names:
        #        raise ValueError(
        #            f"{name} is not a model parameter. All placeholder variables should correspond to model "
        #            f"parameters."
        #        )

        #    if name in self._name_to_variable.keys():
        #        raise ValueError(
        #            f"{name} is already a registered placeholder variable with shape "
        #            f"{self._name_to_variable[name].type.shape}"
        #        )

        # Store with the unprefixed name as the key - prefixing happens when looking up in PyMC model
        self._name_to_variable[name] = variable

        # Next, use these symbolic variables to build the statespace matrices by assigning each parameter
        # to its correct location in the correct matrix
        self.ssm["initial_state", :] = x0
        self.ssm["initial_state_cov", :, :] = P0

        ## To do: implement the no vision trials.
        transition = pt.zeros((self.numTrials, self.k_states, self.k_states))
        transition_vision = pt.stack(
            [pt.stack([self.A, -self.B]), pt.stack([self.A, -self.B])]
        )
        # transition_no_vision = pt.stack([
        #    pt.stack([self.A, 0]),
        #    pt.stack([self.A, 0])
        # ])
        transition = transition[:, :, :].set(transition_vision)
        # transition = transition[self.no_vision_indices, :,:].set(transition_no_vision)
        self.ssm["transition"] = transition

        self.ssm["selection", :, :] = pt.stack([pt.stack([1, 0]), pt.stack([1, 1])])

        self.ssm["state_cov", :, :] = pt.stack(
            [pt.stack([self.var_eta, 0]), pt.stack([0.0, self.var_epsilon])]
        )

        state_intercept = pt.zeros((self.numTrials, self.k_states))
        state_intercept = state_intercept[:, 1].set(self.perturbation)
        self.ssm["state_intercept"] = state_intercept

        self.ssm["design", :, :] = np.array([[0, 1]])

    @property
    def param_names(self):
        base_names = ["x0", "P0", "A", "B", "var_eta", "var_epsilon"]
        # If the model has a name, return prefixed parameter names
        if self.name:
            return [f"{self.name}_{name}" for name in base_names]
        return base_names

    @property
    def state_names(self):
        return ["x", "y"]

    @property
    def observed_states(self):
        return ["y_est"]

    @property
    def shock_names(self):
        return ["var_eta", "var_epsilon"]


def build_one_rate_model(I, Y, P, V, A, B, var_eta, var_epsilon, numTrials):
    I = 0  # To do: This should be replaced by the iterator from the function, but I don't know how I can access the pytensor element.
    tsm = OneRateModel(
        name=f"S_{I}",
        perturbation=P,
        vision=V,
        A=A,
        B=B,
        var_eta=var_eta,
        var_epsilon=var_epsilon,
        numTrials=numTrials,
    )
    tsm.build_statespace_graph(data=Y)


numTestSubjects = 3
numTrials = np.shape(y_obs)[0] + 1

coords = {
    "state": ["x", "y"],
    "state_aux": ["x", "y"],
    "observed_state": ["y_est"],
    "observed_state_aux": ["y_est"],
    "shock": ["var_eta", "var_epsilon"],
    "shock_aux": ["var_eta", "var_epsilon"],
    "subjects": np.arange(numTestSubjects),
}

Y = np.transpose(
    np.concatenate((np.zeros((1, numTestSubjects)), y_obs[:, 0:numTestSubjects]))
)
Y = pt.as_tensor_variable(Y)
V = np.transpose(
    np.concatenate((np.zeros((1, numTestSubjects)), vision[:, 0:numTestSubjects]))
)
V = pt.as_tensor_variable(V)
P = np.transpose(
    np.concatenate((np.zeros((1, numTestSubjects)), perturbation[:, 0:numTestSubjects]))
)
P = pt.as_tensor_variable(P)

with pm.Model(coords=coords) as mod:
    x0 = pm.Data("x0", np.zeros(2, dtype="float"))
    P0 = pm.Data("P0", np.eye(2) * 1)
    A_logit_mu = pm.Normal("A_logit_mu", mu=0, sigma=1)
    A_logit_sd = pm.Gamma("A_logit_sd", mu=1, sigma=1)
    B_logit_mu = pm.Normal("B_logit_mu", mu=0, sigma=1)
    B_logit_sd = pm.Gamma("B_logit_sd", mu=1, sigma=1)

    # ---------- A parameters -------
    A_logit = pm.Normal("A_logit", mu=A_logit_mu, sigma=A_logit_sd, dims=["subjects"])
    A = pm.Deterministic("A", pm.math.sigmoid(A_logit))

    # ---------- B parameters -------
    B_logit = pm.Normal("B_logit", mu=B_logit_mu, sigma=B_logit_sd, dims=["subjects"])
    B = pm.Deterministic("B", pm.math.sigmoid(B_logit))

    # ---------- noise -------
    var_total = pm.Gamma("var_total", mu=4, sigma=3, dims=["subjects"])
    ratio_logit = pm.Normal("ratio_logit", mu=0, sigma=1, dims=["subjects"])
    ratio = pm.Deterministic("ratio", pm.math.sigmoid(ratio_logit))
    var_eta = pm.Deterministic("var_eta", var_total * ratio)
    var_epsilon = pm.Deterministic("var_epsilon", var_total * (1 - ratio))

    pytensor.scan(
        build_one_rate_model,
        sequences=[pt.arange(numTestSubjects), Y, P, V, A, B, var_eta, var_epsilon],
        non_sequences=[numTrials],
    )

prior_checks = pm.sample_prior_predictive(samples=1000, model=mod)

az.plot_posterior(
    prior_checks, var_names=["A", "B", "var_eta", "var_epsilon"], group="prior"
)
az.plot_posterior(
    prior_checks, var_names=["A_logit", "A", "B_logit", "B"], group="prior"
)
az.plot_posterior(prior_checks, var_names=["var_eta", "var_epsilon"], group="prior")
az.plot_ppc(prior_checks, observed=True, group="prior")

idata = pm.sample(
    1000,
    tune=1000,
    target_accept=0.95,
    model=mod,
    nuts_sampler="nutpie",
    nuts_sampler_kwargs={"backend": "jax", "gradient_backend": "jax"},
)

pm.sample_posterior_predictive(
    idata, model=mod, extend_inferencedata=True, var_names=["obs"]
)

idata

var_names = ["A", "B", "var_eta", "var_epsilon"]
az.summary(data=idata, var_names=var_names, round_to=2, hdi_prob=0.95)
az.plot_trace(idata, var_names=var_names)
plt.tight_layout(pad=1.0)
plt.show()
az.plot_posterior(idata, var_names=var_names)
az.to_netcdf(idata, "Output/model2state1subj.nc")
idata.posterior_predictive["obs"].mean(["chain", "draw"])
fig = plt.figure()
plt.plot(
    idata.posterior_predictive["time"],
    idata.posterior_predictive["obs"].mean(["chain", "draw"]),
    label="Mean posterior predictive",
)
plt.plot(Y, label="Observed data")
plt.legend(frameon=False)
