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
    #Define systems dynamic
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
    opti.subject_to(g_OP[0] <= (p[5]+p[7]))#limit gain increase
    opti.subject_to(g_OP[0] >= (p[5]-p[7]))#limit gain decrease
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

def MPC_simulation(x,g,A,G,N,nrT,X0,w,g_max,h,g_eq,x_s,delta_g):

    f = probability_ode(x,g,A,G)
    opti = cas.Opti()
    x_OP = opti.variable(N,1)
    g_OP = opti.variable(N,1)
    p = opti.parameter(8,1) #w, x_s, x0, g_max, h, g_p, g_s, delta_g
    MPC_problem_formulation(opti,N,f,x_OP,g_OP,p)
    opts = {'ipopt.print_level':0, 'print_time':0}
    opti.solver('ipopt',opts)

    #Non-linear MPC loop
    x_MPC_value = np.zeros(nrT)
    g_MPC_value = np.zeros(nrT)
    g_temp = np.zeros(N)
    x_MPC_value[0] = X0
    F = probability_ode(x,g,A,G)
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
            #print('eq.reached for k =',(k+1), 'with h =', h)  
            index_eq = k+1
            q = 1     

    return x_MPC_value, g_MPC_value,index_eq

def uncontrolled_dynamic(nrT,X0,x,g,A,G,h):
    x_euler_values = np.zeros(nrT)
    x_euler_values[0] = X0
    F = probability_ode(x,g,A,G) #g is 0, but this avoind multiplying =*empty matrix G
    for k in range(0, nrT-1):
        x_euler_values[k+1] = x_euler_values[k] + h*F(x_euler_values[k],0)
    return x_euler_values



