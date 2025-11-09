#02/10/2025 Giulia B
#Simulation of the dynamic system described in "Equilibrium Selection in Replicator Equations
#Using Adaptive-Gain Control" with MPC control control

import casadi as cas
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import function_file as ff

#variable declaration
A = np.zeros((2,2))
A[0,0] = 1 #a
A[0,1] = 2   #b
A[1,0] = 0.5 #c
A[1,1] = 1.5 #d

G3 = np.zeros((2,2))
G3[1,0] = 1 #G3
G4 = np.zeros((2,2))
G4[1,1] = 1 #G4
G = np.zeros((2,2))
#G[0,1] = 1 #G2
G_seq = [G3,G4]

alpha = A[0,0]-A[1,0] #a - c 
beta = A[0,1]-A[1,1] #b - d 
x_system = -beta/(-alpha-beta) #equilibrium for anticorodination games

T = 40 #interval 0-T
X0 = [0.8, 0.4] #initial condition
N = 10 #MPC horizon
x_s = [0.4,0.8] #desired equilibrium
#gain at the equilibrium
#g_eq = alpha + beta*((1-x_s)/x_s) #G3
#g_eq = beta - ((alpha*x_s)/(x_s-1)) #G4
#g_eq = ((alpha - beta)*x_s+beta)/(x_s-1) #G2 


#Non linear MPC parameters - cost function l = sum(over N) w*(x-x_s)^2+g^2
w = 50
g_max = 5 
delta_g = [0.5,1,g_max]
gA = A + (G*g_max)

#step calculation
h = ff.step_calculation(gA) 
nrT = math.ceil(T/h)

#state variable - 1D system
x = cas.MX.sym('x')

#control - single control to be optimized
g = cas.MX.sym('g')

#dynamical system - dx = f(x,u)
data =[]

for j in range(len(G_seq)):
    for k in range(len(X0)):
        if j == 0:
            g_eq = alpha + beta*((1-x_s[k])/x_s[k])
        else:
            g_eq = beta - ((alpha*x_s[k])/(x_s[k]-1))

        for i in range(len(delta_g)):
            x_MPC_value, g_MPC_value, index_eq = ff.MPC_simulation(x,g,A,G_seq[j],N,nrT,X0[k],w,g_max,h,g_eq,x_s[k],delta_g[i])
            max_gain, gain_integral = ff.gain_info(g_MPC_value, index_eq, h)
            #print(i,",",X0[k],",", x_s[k],",",delta_g[m],",",eq_index,",",max_gain,",",gain_integral, file=data_file)
            print(j," | ",X0[k]," | ", x_s[k]," | ",delta_g[i]," | ",index_eq," | ",max_gain," | ",gain_integral)
            data.append(x_MPC_value[0:(nrT-1)])
            data.append(g_MPC_value[0:(nrT-1)])


x_euler_values = np.zeros(nrT)
x_euler_values[0] = X0[0]
F = ff.probability_ode(x,g,A,G)

for k in range(0, nrT-1): #nrT-1
    #controlled dynamic
    #uncontrolled dynamic
    x_euler_values[k+1] = x_euler_values[k] + h*F(x_euler_values[k],0)




#data analisys

g_normalized = data[1]/g_max

#plotting
time = np.linspace(0, T, (nrT-1))
plt.figure(figsize=(12, 6))
plt.xlabel('time')
plt.ylabel('state,gain')
plt.plot(time, x_euler_values[0:(nrT-1)], label='Uncontrolled state', linestyle='--', color='blue',linewidth=3)
plt.plot(time, data[0], label='Controlled state',color='orange',linewidth=3)
plt.plot(time, g_normalized[0:(nrT-1)], label='Gain', color='red',linewidth=3 )
plt.legend(loc= 'upper right')
#plt.savefig(path_and_name, format = 'png')
plt.show()


