#02/10/2025 Giulia B
#Simulation of the dynamic system described in "Equilibrium Selection in Replicator Equations
#Using Adaptive-Gain Control" without control

from casadi import *
import numpy as np
import math
import matplotlib.pyplot as plt
import function_file as ff


#variable declaration
x = SX.sym("x") #probability
A = np.zeros((2,2))
A[0,0] = 1
A[0,1] = 3
A[1,0] = 2
A[1,1] = 1

#System initialization
T = 10 #interval 0-T
X0 = 0.8 #initial condition


#step calculation
#estimation to be found - 2 options currently
h = 0.1
nrT = math.ceil(T/h)


#Euler integration step
def euler_step(F,xk, h ):
    step = F(xk, 0)
    xk_next = xk + h*step
    return xk_next


def probability_ode_without_control(x,A):
    r = [A[0,0]*x + A[0,1]*(1-x), A[1,0]*x + A[1,1]*(1-x)]
    dx = x*(1-x)*(r[0]-r[1]) #replicator equation 
    F = Function('F',[x],[dx],['x'],['ode'])
    return F

#dynamic uncontrolled system
x_euler_values = np.zeros(nrT)
x_euler_values[0] = X0
time = np.linspace(0, T, nrT)
F = probability_ode_without_control(x,A)
for k in range(nrT-1):
    x_euler_values[k+1] = euler_step(F,x_euler_values[k], h,)


#plotting
time = np.linspace(0, T, (nrT-1))
plt.figure(figsize=(12, 6))
plt.rcParams.update({'font.size': 15})
plt.xlabel('time',fontsize = 17)
plt.ylabel('state',fontsize = 17)
plt.ylim(0,1)
plt.plot(time, x_euler_values, label='Uncontrolled state - Euler Method', linestyle='-')
plt.legend(loc= 'best',fontsize = 17)
plt.grid(color = 'lightgray')
plt.show()

