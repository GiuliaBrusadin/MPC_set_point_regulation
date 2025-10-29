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
w = 1000
g_max = 20 

#step calculation
gA = A + (G*g_max)
print(gA)
MAX = 0.0
for i in range(0,2):
    for k in range(0,2):
        print(i,k)
        if MAX < gA[i,k]:
            MAX = gA[i,k]

print(MAX)
#nrT = math.ceil(T/h)
