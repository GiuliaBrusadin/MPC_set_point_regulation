#this file contains the function used in code
import casadi as cas
import numpy as np
import math
import pandas as pd

def step_calculation(M):
    #step calculation using MAX payoff as Lipschitz constant - M 2x2 matrix
    MAX = 0.0
    for i in range(0,2):
        for k in range(0,2):
            if MAX < M[i,k]:
             MAX = M[i,k]
    return (2.0/MAX) 

def probability_ode(x,g,A,G):
    gA = A + (G*g)
    r = [gA[0,0]*x + gA[0,1]*(1-x), gA[1,0]*x + gA[1,1]*(1-x)]
    dx = x*(1-x)*(r[0]-r[1]) #replicator equation 
    F = cas.Function('F',[x,g],[dx],['x','g'],['ode'])
    return F

def MPC_problem_formulation(opti,N,f,x_OP,g_OP,p):
     #define cost function and constraints
    def cost_function(x_OP,g_OP,p):
        #since sum() is not accepted, do it in matrix form
        P2 = p[1]*cas.DM.ones(N) #vector of equilibrium state
        P7 = p[6]*cas.DM.ones(N) #vector of equilibrium gain
        return p[0]*cas.mtimes(cas.transpose(x_OP-P2),(x_OP-P2))+cas.mtimes(cas.transpose(g_OP-P7),(g_OP-P7))

    opti.minimize(cost_function(x_OP,g_OP,p))

    for k in range(0, N-2):
        #discretized dynamics using Euler'e method - x (k+1) = x(k)+h*f(x(k),u(k))
        opti.subject_to(x_OP[k+1] == x_OP[k] + p[4]*f(x_OP[k],g_OP[k]))
    opti.subject_to(x_OP[0] == p[2])
    opti.subject_to(g_OP[:] > 0)
    opti.subject_to(g_OP[:] < p[3])
    opti.subject_to(g_OP[0] <= (p[5]+p[7])) #limit gain increase
    opti.subject_to(x_OP[N-1] == p[1]) #terminal constraints
    
    return opti, p


def data_print_to_csv(uncontrolled_state, controlled_state, control, name_and_path):
    data = np.column_stack([uncontrolled_state, controlled_state, control])
    DF = pd.DataFrame(data)
    DF.to_csv(name_and_path)

def gain_info(control, index_eq, h):
    max_gain = np.max(control)
    gain_integral = 0.0
    for k in range(index_eq):
        gain_integral += h*control[k]
    
    return max_gain, gain_integral



