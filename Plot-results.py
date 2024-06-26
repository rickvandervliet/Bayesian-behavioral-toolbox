from os.path import dirname, realpath, join as pjoin
import arviz as az
import matplotlib.pyplot as plt

dir_path = dirname(realpath(__file__))
posterior_fname = pjoin(dir_path, 'posteriorDatav2.nc')
idata = az.from_netcdf(posterior_fname)

az.style.use("arviz-doc")

az.plot_trace(idata, var_names=("A1mu", "A1std"))
plt.show()

az.plot_trace(idata, var_names=("B1mu", "B1std"))
plt.show()

az.plot_trace(idata, var_names=("etamu", "etastd"))
plt.show()

az.plot_trace(idata, var_names=("epsilonmu", "epsilonstd"))
plt.show()

ax = az.plot_forest(
    idata,
    var_names=["A1"],
    combined=False,
    figsize=(11.5, 5),
    colors="C1",
)
plt.show()

ax = az.plot_forest(
    idata,
    var_names=["B1"],
    combined=False,
    figsize=(11.5, 5),
    colors="C1",
)
plt.show()

ax = az.plot_forest(
    idata,
    var_names=["sigma_eta"],
    combined=False,
    figsize=(11.5, 5),
    colors="C1",
)
plt.show()

ax = az.plot_forest(
    idata,
    var_names=["sigma_epsilon"],
    combined=False,
    figsize=(11.5, 5),
    colors="C1",
)
plt.show()