import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import scipy as sp
from scipy.integrate import odeint
import numpy as np
from scipy.stats import qmc

#Parameter vector
params = np.array([lambda_p, lambda_c, lambda_i, Beta, alpha_c, alpha_i, S_pc, S_pn, S_i, r, n, x_0[0], x_0[1], x_0[2]])

#Latin Hypercube
sampler = qmc.LatinHypercube(d=(len(params)))
sample = sampler.random(n=100)
#print(f'Sample before scaling is {sample}')
uni_scaler = 20

# new goal, add initial conditions into our LHS stuff.  
l_bounds = [lambda_p - uni_scaler*lambda_p, lambda_c - uni_scaler*lambda_c, lambda_i - uni_scaler*lambda_i, Beta - uni_scaler*Beta, alpha_c - uni_scaler*alpha_c, alpha_i - uni_scaler*alpha_i, S_pc - uni_scaler*S_pc, S_pn - uni_scaler*S_pn,  S_i - uni_scaler* S_i, r - uni_scaler*r, n - uni_scaler*n, x_0[0] - uni_scaler*x_0[0], x_0[1] - uni_scaler*x_0[1], x_0[2] - uni_scaler*x_0[2]]
u_bounds = [lambda_p + uni_scaler*lambda_p, lambda_c + uni_scaler*lambda_c, lambda_i + uni_scaler*lambda_i, Beta + uni_scaler*Beta, alpha_c + uni_scaler*alpha_c, alpha_i + uni_scaler*alpha_i, S_pc + uni_scaler*S_pc, S_pn + uni_scaler*S_pn,  S_i + uni_scaler* S_i, r + uni_scaler*r, n + uni_scaler*n, x_0[0] + uni_scaler*x_0[0], x_0[1] + uni_scaler*x_0[1], x_0[2] + uni_scaler*x_0[2]]
scaled_sample = qmc.scale(sample, l_bounds, u_bounds)
#print(f'Scaled LHS sample is {scaled_sample}')

SSEs = np.zeros(len(scaled_sample))


for i in range(0,len(scaled_sample)):
    lambda_p   = scaled_sample[i,0] 
    lambda_c   = scaled_sample[i,1]
    lambda_i   = scaled_sample[i,2]
    Beta       = scaled_sample[i,3]
    alpha_c    = scaled_sample[i,4]
    alpha_i    = scaled_sample[i,5]
    S_pc       = scaled_sample[i,6]
    S_pn       = scaled_sample[i,7]
    S_i        = scaled_sample[i,8]
    r          = scaled_sample[i,9]
    n          = scaled_sample[i,10]
    x_0        = [scaled_sample[i,11],scaled_sample[i,12],scaled_sample[i,13], 0, 0]
    full_sim = odeint(odes,x_0,t)
    C = full_sim[:,1]+full_sim[:,4] # sum of all cancer cells
    C_sim = np.interp(x_coords, t, C)
    ERRS = C_sim - np.array(y_coords)
    SSEs[i] = np.sum(ERRS**2)


#Find the minimum SSE
best_SSE = min(SSEs[0:i])

#Find the index of the minimum SSE
best_idx = np.argmin(SSEs)

#Exact parameter values for that index
best_lambda_p   = scaled_sample[best_idx, 0]
best_lambda_c   = scaled_sample[best_idx, 1]
best_lambda_i   = scaled_sample[best_idx, 2]
best_Beta       = scaled_sample[best_idx, 3]
best_alpha_c    = scaled_sample[best_idx, 4]
best_alpha_i    = scaled_sample[best_idx, 5]
best_S_pc       = scaled_sample[best_idx, 6]
best_S_pn       = scaled_sample[best_idx, 7]
best_S_i        = scaled_sample[best_idx, 8]
best_r          = scaled_sample[best_idx, 9]
best_n          = scaled_sample[best_idx, 10]

print(f"The smallest SSE is {best_SSE}")
print(f"Best parameters at index {best_idx}:")
print(f"lambda_p   = {best_lambda_p}")
print(f"lambda_c   = {best_lambda_c}")
print(f"lambda_i   = {best_lambda_i}")
print(f"Beta       = {best_Beta}")
print(f"alpha_c    = {best_alpha_c}")
print(f"alpha_i    = {best_alpha_i}")
print(f"S_pc       = {best_S_pc}")
print(f"S_pn       = {best_S_pn}")
print(f"S_i        = {best_S_i}")
print(f"r          = {best_r}")
print(f"n          = {best_n}")