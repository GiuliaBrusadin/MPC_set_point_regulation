#31/10/2025 Giulia B
#Simulation of the dynamic system described in "Equilibrium Selection in Replicator Equations
#Using Adaptive-Gain Control" with originally thoerized control

import casadi as cas
import numpy as np
import math
import matplotlib.pyplot as plt

#variable declaration
A = np.zeros((2,2))
A[0,0] = 1 #a
A[0,1] = 3 #b
A[1,0] = 0 #c
A[1,1] = 2 #d

G = np.zeros((2,2))
G[1,0] = 1 #G3
#G[0,1] = 1 #G2

alpha = 0.2#A[0,0]-A[1,0] #a - c
beta = 0.3#A[0,1]-A[1,1] #b - d

T = 100 #interval 0-T
X0 = 0.8 #initial condition
G0 = 0.2 #initial gain
N = 5 #MPC horizon
x_s = 0.4 #desired equilibrium
p = 0.1
g_sd = (alpha*x_s-beta*x_s+beta)/x_s#gain at the equilibrium for dominant srategy
g_sa = A[0,0]+A[1,1]-A[1,0]-A[0,1]+(beta/x_s) #gain at the equilibrium for anticoordination games
print(g_sd)

#step calculation
#using MAX payoff as Lipschitz constant - this si not relevant to this usecase, 
#but it allows to keep all simulations with the same step to help comparison
g_max = 20
gA = A + (G*g_max)
MAX = 0.0
for i in range(0,2):
    for k in range(0,2):
        if MAX < gA[i,k]:
            MAX = gA[i,k]
h = 2.0/MAX 
#print(h)
nrT = math.ceil(T/h)

#state variables - probability and gain
x = cas.MX.sym('x',1,1)

#control - single control to be optimized
g = cas.MX.sym('g',1,1)


def ode_dynamics(x,g, g_s,x_s,alpha,beta,p):
    dx = +beta*(1/x_s)*x*(1-x)*(x_s-x)-(x*x)*(1-x)*(g-g_s) #dynamic based on replicator equation 
    dg =  g*p*(x-x_s) #control theoretical
    x_dot = cas.vertcat(dx,dg)
    F = cas.Function('F',[x,g],[dx,dg],['x','g'],['dx','dg'])
    print(F)
    return F

x_values = np.zeros(nrT)
g_values = np.zeros(nrT)
x_values[0] = X0
g_values[0] = G0
#step= np.zeros(2)
F = ode_dynamics(x,g,g_sd,x_s,alpha,beta,p)
for i in range(nrT-1):
    step = F(x_values[i], g_values[i])
    # print(step)
    # print(step[1])
    x_values[i+1] = x_values[i] + h*step[0]
    g_values[i+1] = g_values[i] + h*step[1]


# #plotting
time = np.linspace(0,T,(nrT-2))
plt.figure(figsize=(12, 6))
plt.plot(time, x_values[1:(nrT-1)], label='x controlled',color='blue')
plt.plot(time, g_values[1:(nrT-1)], label='Gain', color='red' )
plt.show()
