#02/10/2025 Giulia B
#Simulation of the dynamic system described in "Equilibrium Selection in Replicator Equations
#Using Adaptive-Gain Control" with MPC control control

import casadi as cas
import numpy as np
import math
import matplotlib.pyplot as plt

#variable declaration
A = np.zeros((2,2))
A[0,0] = 1.5   #a
A[0,1] = 2   #b
A[1,0] = 2    #c
A[1,1] = 1    #d

G = np.zeros((2,2))
G[1,0] = 1 #G3
#G[0,1] = 1 #G2
#G[1,1] = 1 #G4

alpha = A[0,0]-A[1,0] #a - c 
beta = A[0,1]-A[1,1] #b - d 
x_system = -beta/(-alpha-beta) #equilibrium for anticorodination games

T = 40 #interval 0-T
X0 = 0.1 #initial condition
N = 10 #MPC horizon
x_s = 0.3 #desired equilibrium
#gain at the equilibrium
g_eq = alpha + beta*((1-x_s)/x_s) #G3
#g_eq = beta - ((alpha*x_s)/(x_s-1)) #G4
#g_eq = ((alpha - beta)*x_s+beta)/(x_s-1) #G2 - dominant strategy brings to xeq = 0


#Non linear MPC parameters - cost function l = sum(over N) w*(x-x_s)^2+g^2
w = 1000
g_max = 5 

#step calculation
#using MAX payoff as Lipschitz constant
gA = A + (G*g_max)
MAX = 0.0
for i in range(0,2):
    for k in range(0,2):
        if MAX < gA[i,k]:
            MAX = gA[i,k]
h = 2.0/MAX 
#print(h)
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
p = opti.parameter(7,1) #w, x_s, x0, g_max, h, g_p, g_s

def cost_function(x_OP,g_OP,p):
    #since sum() is not accepted, do it in matrix form
    P2 = p[1]*cas.DM.ones(N) #vector of equilibrium state
    P7 = p[6]*cas.DM.ones(N) #vector of equilibrium gain
    return p[0]*cas.mtimes(cas.transpose(x_OP-P2),(x_OP-P2))+cas.mtimes(cas.transpose(g_OP-P7),(g_OP-P7))

opti.minimize(cost_function(x_OP,g_OP,p))

for k in range(0, N-2):
    f = probability_ode(x,g,A,G)
    #discretized dynamics using Euler'e method - x (k+1) = x(k)+h*f(x(k),u(k))
    opti.subject_to(x_OP[k+1] == x_OP[k] + p[4]*f(x_OP[k],g_OP[k]))
opti.subject_to(x_OP[0] == p[2])
opti.subject_to(g_OP[:] > 0)
opti.subject_to(g_OP[:] < p[3])
#opti.subject_to((p[5]-2) <= g_OP[0])
#opti.subject_to(g_OP[0] <= (p[5]+0.5)) #limit gain increase
opti.subject_to(x_OP[N-1] == p[1]) #terminal constraints
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
    g_prec = g_MPC_value[k-1]
    opti.set_value(p,[w,x_s,x_MPC_value[k], g_max,h,g_prec,g_eq])
    sol = opti.solve()
    g_temp = sol.value(g_OP)
    g_MPC_value[k] = g_temp[0]
    x_MPC_value[k+1] = x_MPC_value[k] + h*F(x_MPC_value[k],g_MPC_value[k])
    #print(x_s - x_MPC_value[k+1])
    if( q==0 and x_s ==(float('%.3f'%(x_MPC_value[k+1])))):
       print('eq.reached for k =',(k+1), 'with h =', h)  
       q = 1     
    #uncontrolled dynamic
    x_euler_values[k+1] = x_euler_values[k] + h*F(x_euler_values[k],0)


g_normalized = g_MPC_value/g_max
print('max gain:',np.max(g_MPC_value))
print('final uncontrolled x:',x_euler_values[(nrT-1)])
#print('final controlled x:',x_MPC_value[(nrT-1)])
print('gain at equilibrium',g_eq)
name = ("ac_G3_"+("%s"%g_max)+"_"+("%s"%w)+"_dgnull")
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




