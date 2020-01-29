#!/usr/bin/env python3.7
from __future__ import print_function,division
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import os
import sys
from mpl_toolkits.mplot3d import Axes3D

hdf = pd.HDFStore('NNLO_templates.h5', mode='r')
np.random.seed(1)

obs = "ptll"
mt = 172.5
data = hdf[obs]
nom = data[data.mt == 172.5]


X = data[obs].to_numpy()
Xmt = data[[obs,'mt']].to_numpy()
y = (data["binContent"]/data["scaling"]).to_numpy()
dy = (data["binError"]/data["scaling"]).to_numpy()



#y /= 100
#dy /= 100

#dy[5] *= 10
#dy[6] *= 15

#X = np.atleast_2d(X).T



# TODO: Try different kernels
kernel = C(1.0, (1e-3, 1e3)) * RBF([10,1], (1e-2, 1e2))

# alpha: variance at each datapoint
# normalize_y = True  produces better agreement with input template data 
gp = GaussianProcessRegressor(kernel=kernel, alpha=dy**2,
                              n_restarts_optimizer=10, normalize_y=True)

gp.fit(Xmt, y)

# Domain to evaluate interpolation at
xmin = nom[obs].iloc[0]  - nom["binWidth2"].iloc[0]
xmax = nom[obs].iloc[-1] + nom["binWidth2"].iloc[-1]


#x = np.atleast_2d(np.linspace(xmin, xmax, 1000)).T
mtmin = 170.
mtmax = 175.
deltaMT = 0.1
Nmt = int( (mtmax-mtmin)/deltaMT + 1)

Nobs = 200
obspoints = np.tile(np.linspace(xmin, xmax, Nobs), Nmt)
mtpoints = np.repeat(np.linspace(mtmin,mtmax,Nmt), Nobs) 

x = np.column_stack( (obspoints, mtpoints) )

# Do interpolation
y_pred, sigma = gp.predict(x, return_std=True)
sigma *= 1.68

#binCenters = np.tile(X, Nmt)
#mtmorphpoints = np.repeat(np.linspace(mtmin,mtmax,Nmt), len(X)) 
#xmorph = np.column_stack( (binCenters, mtmorphpoints) )
#
#
#ymorph, sigmamorph = gp.predict(xmorph, return_std=True)
#sigmamorph *= 1.6800




# Plot result
fig = plt.figure()
ax = Axes3D(fig)


# Known points
#plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label='Observations')

# Interpolated points
#plt.plot(x, y_pred, 'b-', label='Prediction')
ax.plot(Xmt.T[0], Xmt.T[1], y, 'r.', markersize=10, label='Observations')
ax.plot(x.T[0], x.T[1], y_pred, 'b-', label='Prediction')


#plt.fill(np.concatenate([x, x[::-1]]), \
#         np.concatenate([y_pred - 1.6800 * sigma,\
#                        (y_pred + 1.6800 * sigma)[::-1]]),\
#         alpha=.5, fc='b', ec='None', label='68% confidence interval')

ax.legend(loc='upper right')
ax.set_title("%s NNLO" % obs)
ax.set_xlabel("%s [GeV]" % obs)
ax.set_ylabel("$m_{t}$ [GeV]")
ax.set_zlabel("Events")

plt.show()


