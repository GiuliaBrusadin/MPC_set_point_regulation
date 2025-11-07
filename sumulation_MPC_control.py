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
A[0,1] = 1   #b
A[1,0] = 1.3 #c
A[1,1] = 0.7 #d

G = np.zeros((2,2))
G[1,0] = 1 #G3
#G[0,1] = 1 #G2
#G[1,1] = 1 #G4

alpha = A[0,0]-A[1,0] #a - c 
beta = A[0,1]-A[1,1] #b - d 
x_system = -beta/(-alpha-beta) #equilibrium for anticorodination games

T = 40 #interval 0-T
X0 = 0.8 #initial condition
N = 10 #MPC horizon
x_s = 0.4 #desired equilibrium
#gain at the equilibrium
g_eq = alpha + beta*((1-x_s)/x_s) #G3
#g_eq = beta - ((alpha*x_s)/(x_s-1)) #G4
#g_eq = ((alpha - beta)*x_s+beta)/(x_s-1) #G2 


#Non linear MPC parameters - cost function l = sum(over N) w*(x-x_s)^2+g^2
w = 10
g_max = 5 
delta_g = 0.5
gA = A + (G*g_max)

#step calculation
h = ff.step_calculation(gA) 
nrT = math.ceil(T/h)

#state variable - 1D system
x = cas.MX.sym('x')

#control - single control to be optimized
g = cas.MX.sym('g')

#dynamical system - dx = f(x,u)

f = ff.probability_ode(x,g,A,G)
opti = cas.Opti()
x_OP = opti.variable(N,1)
g_OP = opti.variable(N,1)
p = opti.parameter(8,1) #w, x_s, x0, g_max, h, g_p, g_s, delta_g
ff.MPC_problem_formulation(opti,N,f,x_OP,g_OP,p)
opts = {'ipopt.print_level':0, 'print_time':0}
opti.solver('ipopt',opts)

#Non-linear MPC loop
x_MPC_value = np.zeros(nrT)
g_MPC_value = np.zeros(nrT)
g_temp = np.zeros(N)
x_MPC_value[0] = X0
x_euler_values = np.zeros(nrT)
x_euler_values[0] = X0
F = ff.probability_ode(x,g,A,G)
q = 0
index_eq = 0

for k in range(0, nrT-1): #nrT-1
    #controlled dynamic
    g_prec = g_MPC_value[k-1]
    opti.set_value(p,[w,x_s,x_MPC_value[k], g_max,h,g_prec,g_eq,delta_g])
    sol = opti.solve()
    g_temp = sol.value(g_OP)
    g_MPC_value[k] = g_temp[0]
    x_MPC_value[k+1] = x_MPC_value[k] + h*F(x_MPC_value[k],g_MPC_value[k])
    #print(x_s - x_MPC_value[k+1])
    if( q==0 and x_s ==(float('%.3f'%(x_MPC_value[k+1])))):
       print('eq.reached for k =',(k+1), 'with h =', h)  
       index_eq = k+1
       q = 1     
    #uncontrolled dynamic
    x_euler_values[k+1] = x_euler_values[k] + h*F(x_euler_values[k],0)




#data analisys

g_normalized = g_MPC_value/g_max
g_MPC_max, gain_integral = ff.gain_info(g_MPC_value, index_eq, h)
print('max gain:',g_MPC_max)
print('final uncontrolled x:',x_euler_values[(nrT-1)])
#print('final controlled x:',x_MPC_value[(nrT-1)])
print('gain at equilibrium',g_eq)
print('gain energy is:', gain_integral)

name = ("ac_G3_"+("%s"%g_max)+"_"+("%s"%w)+"_dg05")
path_and_name = ("/home/giulia/Documents/Evolutionary_GT/Graphs/anti_coordination/ac_G3/ac_G3_x0_lower/"+name+".png")

#plotting
time = np.linspace(0, T, (nrT-2))
plt.figure(figsize=(12, 6))
plt.xlabel('time')
plt.ylabel('state,gain')
plt.plot(time, x_euler_values[1:(nrT-1)], label='Uncontrolled state', linestyle='--', color='blue',linewidth=3)
plt.plot(time, x_MPC_value[1:(nrT-1)], label='Controlled state',color='orange',linewidth=3)
plt.plot(time, g_normalized[1:(nrT-1)], label='Gain', color='red',linewidth=3 )
plt.legend(loc= 'upper right')
#plt.savefig(path_and_name, format = 'png')
plt.show()


