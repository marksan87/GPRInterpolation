#!/usr/bin/env python3.7
from __future__ import print_function,division
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import os
import sys

hdf = pd.HDFStore('NNLO_templates.h5', mode='r')
np.random.seed(1)

obs = "ptll"
mt = 172.5
data = hdf[obs][hdf[obs].mt == mt]

X = data[obs].to_numpy()
#y = data["binContent"].to_numpy()
#dy = data["binError"].to_numpy()

y = (data["binContent"]/data["scaling"]).to_numpy()
dy = (data["binError"]/data["scaling"]).to_numpy()
#y  *= 10000
#dy *= 10000

#y  *= 109023.25
#dy *= 109023.25

#dy[5] *= 10
#dy[6] *= 15

X = np.atleast_2d(X).T

# TODO: Try different kernels
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

# alpha: variance at each datapoint
# normalize_y = True  produces better agreement with input template data 
gp = GaussianProcessRegressor(kernel=kernel, alpha=dy**2,
                              n_restarts_optimizer=10, normalize_y=True)

gp.fit(X, y)

# Domain to evaluate interpolation at
xmin = data[obs].iloc[0]  - data["binWidth2"].iloc[0]
xmax = data[obs].iloc[-1] + data["binWidth2"].iloc[-1]
x = np.atleast_2d(np.linspace(xmin, xmax, 1000)).T

# Do interpolation
y_pred, sigma = gp.predict(x, return_std=True)


# Plot result
plt.figure()

# Known points
plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label='Observations')

# Interpolated points
plt.plot(x, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([x, x[::-1]]), \
         np.concatenate([y_pred - 1.6800 * sigma,\
                        (y_pred + 1.6800 * sigma)[::-1]]),\
         alpha=.5, fc='b', ec='None', label='68% confidence interval')

plt.legend(loc='upper right')
plt.title("%s NNLO  $m_{t}$ = %.1f" % (obs,mt))
plt.xlabel("%s [GeV]" % obs)
plt.ylabel("Events")

plt.show()


