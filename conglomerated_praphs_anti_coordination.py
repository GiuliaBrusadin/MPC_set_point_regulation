#graphs of control and state for dominant strategy games
import casadi as cas
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import function_file as ff

#variable declaration
A = np.zeros((2,2))
A[0,0] = 1.1   #a
A[0,1] = 1.5 #b
A[1,0] = 1.5 #c
A[1,1] = 1   #d

G2 = np.zeros((2,2))
G2[0,1] = 1 #G3
G3 = np.zeros((2,2))
G3[1,0] = 1 #G3

alpha = A[0,0]-A[1,0] #a - c 
beta = A[0,1]-A[1,1] #b - d 
x_system = (A[0,1]-A[1,1])/(A[1,0]-A[0,0]+A[0,1]-A[1,1]) #equilibrium for anticorodination games (0.62)
print("System's uncontrolled equilibrium:",x_system)

#simulation parameters
T = 10 #interval 0-T
N = 10 #MPC horizon
X0 = 0.5
x_s2 = 0.9
x_s3 = 0.2

#Non linear MPC parameters
w = 140
g_max = 5 
delta_g = 1

#step calculation
h2 = ff.step_calculation(A + (G2*g_max))
h3 = ff.step_calculation(A + (G3*g_max))
h = min(h2,h3)
print('timestep h: ', h)
nrT = math.ceil(T/h)

#state variable - 1D system
x = cas.MX.sym('x')

#control - single control to be optimized
g = cas.MX.sym('g')

data_x = []
data_g = []

#G2
g_eq = ((alpha - beta)*x_s2+beta)/(x_s2-1)
#print('G2 eq',g_eq)
x_MPC_value, g_MPC_value, index_eq = ff.MPC_simulation(x,g,A,G2,N,nrT,X0,w,g_max,h,g_eq,x_s2,delta_g)
max_gain, gain_integral = ff.gain_info(g_MPC_value, index_eq, h)
data_x.append(x_MPC_value[0:(nrT-1)])
data_g.append(g_MPC_value[0:(nrT-1)])
#G3
g_eq = alpha + beta*((1-x_s3)/x_s3)
#print('G3 eq',g_eq)
x_MPC_value, g_MPC_value, index_eq = ff.MPC_simulation(x,g,A,G3,N,nrT,X0,w,g_max,h,g_eq,x_s3,delta_g)
max_gain, gain_integral = ff.gain_info(g_MPC_value, index_eq, h)
data_x.append(x_MPC_value[0:(nrT-1)])
data_g.append(g_MPC_value[0:(nrT-1)])

#uncontrolled equilibriums - for plotting
arr = np.ones(nrT)
uncontroled_x = arr*x_system
equilibrium_G2  = x_s2*arr
equilibrium_G3 = x_s3*arr

#Plots
time = np.linspace(0, T, (nrT-1))
plt.figure(figsize=(12, 6))
plt.rcParams.update({'font.size': 15})
plt.xlabel('time',fontsize = 17)
plt.ylim(0,1)
plt.plot(time, uncontroled_x[0:(nrT-1)], label='Uncontrolled equilibrium', linestyle=':', color='purple',linewidth=3)
plt.plot(time, data_x[0], label='Controlled state - G2 ',color='pink',linewidth=3,linestyle='-')
plt.plot(time, data_x[1], label='Controlled state - G3 ',color='fuchsia',linewidth=3,linestyle='--')
plt.plot(time, equilibrium_G2[0:(nrT-1)], label='Desired equilibrium ', linestyle=':', color='grey',linewidth=3)
plt.plot(time, equilibrium_G3[0:(nrT-1)], linestyle=':', color='grey',linewidth=3)
plt.legend(loc= (0.55,0.24),fontsize = 16)
plt.grid(color = 'lightgray')
plt.show()

plt.figure(figsize=(12, 6))
plt.rcParams.update({'font.size': 15})
plt.xlabel('time',fontsize = 17)
plt.ylim(0,5)
plt.plot(time, data_g[0]/max_gain, label='Gain - G2',color='pink',linewidth=3,linestyle='--')
plt.plot(time, data_g[1]/max_gain, label='Gain - G3 ',color='fuchsia',linewidth=3,linestyle='--')
plt.legend(loc= 'upper right',fontsize = 17)
plt.grid(color = 'lightgray')
plt.show()