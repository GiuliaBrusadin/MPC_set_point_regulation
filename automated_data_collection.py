#08/11/2025 Giulia B
#Simulation of the dynamic system described in "Equilibrium Selection in Replicator Equations
#Using Adaptive-Gain Control" with MPC control control - data collction script

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
A[0,0] = 1.5 #a
A[0,1] = 2   #b
A[1,0] = 2   #c
A[1,1] = 1   #d


G1 = np.zeros((2,2))
G1[0,0] = 1 #G3
G2 = np.zeros((2,2))
G2[0,1] = 1 #G3
G3 = np.zeros((2,2))
G3[1,0] = 1 #G3
G4 = np.zeros((2,2))
G4[1,1] = 1 #G4

G = np.zeros((2,2))
#G[0,1] = 1 #G2
G_seq = [G3]

alpha = A[0,0]-A[1,0] #a - c 
beta = A[0,1]-A[1,1] #b - d 
x_system = (A[0,1]-A[1,1])/(A[1,0]-A[0,0]+A[0,1]-A[1,1]) #equilibrium for anticorodination games

T = 10 #interval 0-T
X0 = [0.8, 0.5, 0.1] #initial condition
N = 10 #MPC horizon
x_s = 0.3 #desired equilibrium
#gain at the equilibrium
#g_eq = alpha + beta*((1-x_s)/x_s) #G3
#g_eq = beta - ((alpha*x_s)/(x_s-1)) #G4
#g_eq = ((alpha - beta)*x_s+beta)/(x_s-1) #G2 
#g_eq = beta - alpha - (beta/x_s) #G1


#Non linear MPC parameters - cost function l = sum(over N) w*(x-x_s)^2+g^2
w = 60
g_max = 5 
delta_g = [0.5,1,g_max]

#step calculation

h1 = ff.step_calculation(A + (G3*g_max)) 
h2 = ff.step_calculation(A + (G4*g_max)) 
h = min(h1,h2)
nrT = math.ceil(T/h)

#state variable - 1D system
x = cas.MX.sym('x')

#control - single control to be optimized
g = cas.MX.sym('g')

# data_file = open("./output.txt", "a")
# print("case with G2", file=data_file)
# print("matrix, X0, X_eq,delta_g,step to eq,max_gain,gain integral", file=data_file)
data =[]

for j in range(len(G_seq)):
    for k in range(len(X0)):
        if j == 0:
            g_eq = alpha + beta*((1-x_s)/x_s)
        # else:
        #     g_eq =  beta - alpha - (beta/x_s[k])

        for i in range(len(delta_g)):
            x_MPC_value, g_MPC_value, index_eq = ff.MPC_simulation(x,g,A,G_seq[j],N,nrT,X0[k],w,g_max,h,g_eq,x_s,delta_g[i])
            max_gain, gain_integral = ff.gain_info(g_MPC_value, index_eq, h)
            #print(j,",",X0[k],",", x_s,",",delta_g[i],",",index_eq,",",max_gain,",",gain_integral, file=data_file)
            print(j," | ",X0[k]," | ", x_s," | ",delta_g[i]," | ",index_eq," | ",max_gain," | ",gain_integral)
            data.append(x_MPC_value[0:(nrT-1)])
            data.append(g_MPC_value[0:(nrT-1)])
            print((len(data)-2), ",",(len(data)-1))


# x_euler_values = np.zeros(nrT)
# x_euler_values[0] = X0[1]
# F = ff.probability_ode(x,g,A,G)

# for k in range(0, nrT-1): #nrT-1
#     #controlled dynamic
#     #uncontrolled dynamic
#     x_euler_values[k+1] = x_euler_values[k] + h*F(x_euler_values[k],0)

arr = np.ones(nrT)
x_euler_values = arr*x_system

#plotting
time = np.linspace(0, T, (nrT-1))
plt.figure(figsize=(12, 6))
plt.xlabel('time')
plt.ylabel('state,gain')
plt.plot(time, x_euler_values[0:(nrT-1)], label='Uncontrolled equilibrium', linestyle=':', color='navy',linewidth=3)
plt.plot(time, data[2], label='Controlled state x(0) = 0.8 ',color='blue',linewidth=3,linestyle='-.')
plt.plot(time, data[8], label='Controlled state x(0) = 0.5 ',color='fuchsia',linewidth=3,linestyle='--')
plt.plot(time, data[14], label='Controlled state x(0) = 0.1 ',color='pink',linewidth=3,linestyle='-')
# plt.plot(time, data[2], label='Controlled state - G2',color='orangered',linewidth=3)
# plt.plot(time, data[3]/g_max, label='Gain - G2', color='darkred',linewidth=3, linestyle='--' )
# plt.plot(time, data[14], label='Controlled state - G1',color='lightsalmon',linewidth=3)
# plt.plot(time, data[15]/g_max, label='Gain - G1', color='indianred',linewidth=3, linestyle='--' )
plt.legend(loc= 'best')
plt.grid(color = 'lightgray')
plt.show()


