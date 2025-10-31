#02/10/2025 Giulia B
#Simulation of the dynamic system described in "Equilibrium Selection in Replicator Equations
#Using Adaptive-Gain Control" with MPC control control

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

T = 4 #interval 0-T
X0 = 0.8 #initial condition
N = 5 #MPC horizon
x_s = 0.6 #desired equilibrium

#Non linear MPC parameters - cost function l = sum(over N) w*(x-x_s)^2+g^2
w = 10000
g_max = 20 

#step calculation
#using MAX payoff as Lipschitz constant
gA = A + (G*g_max)
MAX = 0.0
for i in range(0,2):
    for k in range(0,2):
        if MAX < gA[i,k]:
            MAX = gA[i,k]
h = 2.0/MAX 
print(h)
nrT = math.ceil(T/h)

#state variable - 1D system
x = cas.MX.sym('x')

#control - single control to be optimized
g = cas.MX.sym('g')

#dynamical system - dx = f(x,u)
def probability_ode(x,g,A,G):
    gA = A + (G*g)
    r = [gA[0,0]*x + gA[0,1]*(1-x), gA[1,0]*x + gA[1,1]*(1-x)]
    dx = x*(1-x)*(r[0]-r[1]) #replicator equation 
    F = cas.Function('F',[x,g],[dx],['x','g'],['ode'])
    return F

#optimization problem formulation
opti = cas.Opti()
x_OP = opti.variable(N,1)
g_OP = opti.variable(N,1)
p = opti.parameter(5,1) #w, x_s, x0, g_max and h

def cost_function(x_OP,g_OP,p):
    #since sum() is not accepted, do it in matrix form
    P2 = p[1]*cas.DM.ones(N)
    return p[0]*cas.mtimes(cas.transpose(x_OP-P2),(x_OP-P2))+cas.mtimes(cas.transpose(g_OP),g_OP)

opti.minimize(cost_function(x_OP,g_OP,p))

for k in range(0, N-1):
    f = probability_ode(x,g,A,G)
    #discretized dynamics using Euler'e method - x (k+1) = x(k)+h*f(x(k),u(k))
    opti.subject_to(x_OP[k+1] == x_OP[k] + p[4]*f(x_OP[k],g_OP[k]))
opti.subject_to(x_OP[0] == p[2])
opti.subject_to(g_OP[:] > 0)
opti.subject_to(g_OP[:] < p[3])
# opti.subject_to(x_OP[:] > 0)
# opti.subject_to(x_OP[:] < 1)

opts = {'ipopt.print_level':0, 'print_time':0}
opti.solver('ipopt',opts)

#Non-linear MPC loop
x_MPC_value = np.zeros(nrT)
g_MPC_value = np.zeros(nrT)
g_temp = np.zeros(N)
x_MPC_value[0] = X0
x_euler_values = np.zeros(nrT)
x_euler_values[0] = X0
F = probability_ode(x,g,A,G)
q = 0

for k in range(0, nrT-1): #nrT-1
    #controlled dynamic
    opti.set_value(p,[w,x_s,x_MPC_value[k], g_max,h])
    sol = opti.solve()
    g_temp = sol.value(g_OP)
    g_MPC_value[k] = g_temp[0]
    x_MPC_value[k+1] = x_MPC_value[k] + h*F(x_MPC_value[k],g_MPC_value[k])
    print(x_s - x_MPC_value[k+1])
    #uncontrolled dynamic
    x_euler_values[k+1] = x_euler_values[k] + h*F(x_euler_values[k],0)


#plotting
# time = np.linspace(0, T, nrT)
# plt.figure(figsize=(12, 6))
# plt.plot(time, x_euler_values, label='Euler Method', linestyle='--', color='blue')
# plt.plot(time, x_MPC_value, label='NMPC Method',color='orange')
# plt.plot(time, g_MPC_value, label='Gain', color='red' )
# plt.show()





