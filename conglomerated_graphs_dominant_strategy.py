#graphs of control and state for dominant strategy games
import casadi as cas
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import function_file as ff

#variable declaration
A_3_4 = np.zeros((2,2)) # system's equilibrium is 1
A_3_4[0,0] = 1 #a
A_3_4[0,1] = 2 #b
A_3_4[1,0] = 0.5 #c
A_3_4[1,1] = 1.5 #d
A_2_1 = np.zeros((2,2)) # system's equilibrium is 0
A_2_1[0,0] = 0.5 #a
A_2_1[0,1] = 1.5 #b
A_2_1[1,0] = 1 #c
A_2_1[1,1] = 2 #d

G1 = np.zeros((2,2))
G1[0,0] = 1 #G3
G2 = np.zeros((2,2))
G2[0,1] = 1 #G3
G3 = np.zeros((2,2))
G3[1,0] = 1 #G3
G4 = np.zeros((2,2))
G4[1,1] = 1 #G4

G = np.zeros((2,2))

#simulation parameters
T = 10 #interval 0-T
N = 10 #MPC horizon
X0 = 0.8 #initial condition
x_s = 0.4 #desired equilibrium

#Non linear MPC parameters
q = 140
g_max = 5 
delta_g = 1

#step calculation
h1= ff.step_calculation(A_2_1 + (G1*g_max)) 
h2 = ff.step_calculation(A_2_1 + (G2*g_max)) 
h3 = ff.step_calculation(A_3_4 + (G3*g_max)) 
h4 = ff.step_calculation(A_3_4 + (G4*g_max))
h = min(h1,h2,h3,h4)
print('timestep h: ', h)
nrT = math.ceil(T/h)

#state variable - 1D system
x = cas.MX.sym('x')

#control - single control to be optimized
g = cas.MX.sym('g')

data_x = []
data_g = []

#G1 and G2
A = A_2_1
alpha = A[0,0]-A[1,0] #a - c 
beta = A[0,1]-A[1,1] #b - d

#uncontrolled dynamics for system's equilibrium 0
uncontrolled_state_0 = np.zeros(nrT)
uncontrolled_state_0[0] = X0
F = ff.probability_ode(x,g,A,G)
for k in range(0, nrT-1): 
    uncontrolled_state_0[k+1] = uncontrolled_state_0[k] + h*F(uncontrolled_state_0[k],0)

#G1
g_eq = beta - alpha - (beta/x_s)
x_MPC_value, g_MPC_value, index_eq = ff.MPC_simulation(x,g,A,G1,N,nrT,X0,q,g_max,h,g_eq,x_s,delta_g)
max_gain, gain_integral = ff.gain_info(g_MPC_value, index_eq, h)
data_x.append(x_MPC_value[0:(nrT-1)])
data_g.append(g_MPC_value[0:(nrT-1)])

#G2
g_eq = ((alpha - beta)*x_s+beta)/(x_s-1)
x_MPC_value, g_MPC_value, index_eq = ff.MPC_simulation(x,g,A,G2,N,nrT,X0,q,g_max,h,g_eq,x_s,delta_g)
max_gain, gain_integral = ff.gain_info(g_MPC_value, index_eq, h)
data_x.append(x_MPC_value[0:(nrT-1)])
data_g.append(g_MPC_value[0:(nrT-1)])


#G3 e G4
A = A_3_4
alpha = A[0,0]-A[1,0] #a - c 
beta = A[0,1]-A[1,1] #b - d 

#uncontrolled dynamics for system's equilibrium 1
uncontrolled_state_1 = np.zeros(nrT)
uncontrolled_state_1[0] = X0
F = ff.probability_ode(x,g,A,G)
for k in range(0, nrT-1):
    uncontrolled_state_1[k+1] = uncontrolled_state_1[k] + h*F(uncontrolled_state_1[k],0)

#G3
g_eq = alpha + beta*((1-x_s)/x_s)
x_MPC_value, g_MPC_value, index_eq = ff.MPC_simulation(x,g,A,G3,N,nrT,X0,q,g_max,h,g_eq,x_s,delta_g)
max_gain, gain_integral = ff.gain_info(g_MPC_value, index_eq, h)
data_x.append(x_MPC_value[0:(nrT-1)])
data_g.append(g_MPC_value[0:(nrT-1)])
#G4
g_eq = beta - ((alpha*x_s)/(x_s-1))
x_MPC_value, g_MPC_value, index_eq = ff.MPC_simulation(x,g,A,G4,N,nrT,X0,q,g_max,h,g_eq,x_s,delta_g)
max_gain, gain_integral = ff.gain_info(g_MPC_value, index_eq, h)
data_x.append(x_MPC_value[0:(nrT-1)])
data_g.append(g_MPC_value[0:(nrT-1)])

#Plots
time = np.linspace(0, T, (nrT-1))
plt.figure(figsize=(12, 6))
plt.rcParams.update({'font.size': 15})
plt.xlabel('time',fontsize = 17)
plt.ylabel('state',fontsize = 17)
plt.ylim(0,1)
plt.plot(time, uncontrolled_state_0[0:(nrT-1)], label='Uncontrolled state', linestyle=':', color='saddlebrown',linewidth=3)
plt.plot(time, data_x[0], label='Controlled state - G1', linestyle='-', color='chocolate',linewidth=3)
plt.plot(time, data_x[1], label='Controlled state - G2', linestyle='--', color='darkorange',linewidth=3)
plt.legend(loc= (0.55,0.5),fontsize = 17)
plt.grid(color = 'lightgray')
plt.show()

plt.figure(figsize=(12, 6))
plt.rcParams.update({'font.size': 15})
plt.xlabel('time',fontsize = 17)
plt.ylabel('state',fontsize = 17,)
plt.ylim(0,1)
plt.plot(time, uncontrolled_state_1[0:(nrT-1)], label='Uncontrolled state', linestyle=':', color='navy',linewidth=3)
plt.plot(time, data_x[2], label='Controlled state - G3', linestyle='-', color='deepskyblue',linewidth=3)
plt.plot(time, data_x[3], label='Controlled state - G4', linestyle='--', color='blue',linewidth=3)
plt.legend(loc= (0.55,0.5),fontsize = 17)
plt.grid(color = 'lightgray')
plt.show()

plt.figure(figsize=(12, 6))
plt.xlabel('time',fontsize = 17)
plt.ylabel('gain',fontsize = 17,)
plt.rcParams.update({'font.size': 15})
plt.ylim(0,5)
plt.plot(time, data_g[0], label='Gain - G1', linestyle='-', color='chocolate',linewidth=3)
plt.plot(time, data_g[1], label='Gain - G2', linestyle='--', color='darkorange',linewidth=3)
plt.legend(loc= 'upper right',fontsize = 17)
plt.grid(color = 'lightgray')
plt.show()

plt.figure(figsize=(12, 6))
plt.rcParams.update({'font.size': 15})
plt.xlabel('time',fontsize = 17)
plt.ylabel('gain',fontsize = 17,)
plt.ylim(0,5)
plt.plot(time, data_g[2], label='Gain - G3', linestyle='-', color='deepskyblue',linewidth=3)
plt.plot(time, data_g[3], label='Gain - G4', linestyle='--', color='blue',linewidth=3)
plt.legend(loc= 'upper right',fontsize = 17)
plt.grid(color = 'lightgray')
plt.show()