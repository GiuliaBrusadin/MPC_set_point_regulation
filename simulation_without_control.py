#02/10/2025 Giulia B
#Simulation of the dynamic system described in "Equilibrium Selection in Replicator Equations
#Using Adaptive-Gain Control" without control

from casadi import *
import numpy as np
import math
import matplotlib.pyplot as plt


#variable declaration
x = SX.sym("x") #probability
A = np.zeros((2,2))
A[0,0] = 1
A[0,1] = 3
A[1,0] = 2
A[1,1] = 1
G = np.zeros((2,2))
G[1,0] = 1 #G3
G[0,1] = 1 #G2
g = 0

#System initialization
T = 10 #interval 0-T
X0 = 0.8 #initial condition


#step calculation
#estimation to be found - 2 options currently
h = 0.1
nrT = math.ceil(T/h)

#dynamical system
def probability_ode(x,A,G,g):
    gA = A + G*g
    xv = [x , (1-x)]
    r = mtimes(gA, xv)
    dx = x*(1-x)*(r[0,0]-r[1,0])
    return dx


#Euler integration step
def euler_step(xk, h, A, G, g ):
    step = probability_ode(xk, A, G, g)
    xk_next = xk + h*step
    return xk_next

#dynamic uncontrolled system
x_euler_values = np.zeros(nrT)
x_euler_values[0] = X0
time = np.linspace(0, T, nrT)
for k in range(nrT-1):
    x_euler_values[k+1] = euler_step(x_euler_values[k], h, A, G, g)


#plotting
plt.figure(figsize=(12, 6))
plt.plot(time, x_euler_values, label='Euler Method', linestyle='--')
plt.show()

