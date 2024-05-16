import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import arviz as az
plt.style.use('arviz-darkgrid')
x = np.linspace(0, 20, 200)
mean = [0.5]
sigma = [0.5]
for m, s in zip(mean, sigma):
    pdf = st.gamma.pdf(x, (m*m)/(s*s), scale=(s*s)/(m))
    plt.plot(x, pdf, label=r'$\mu$ = {}, $\sigma$ = {}'.format(m, s))
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.legend(loc=1)
plt.show()