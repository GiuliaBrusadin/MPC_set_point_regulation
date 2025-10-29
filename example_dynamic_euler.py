import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81  # gravitational acceleration (m/s^2)
l = 1.0   # length of the pendulum (m)
N = 100
T = 10

# Define symbolic state variables: x[0] is theta, x[1] is theta_dot
x = ca.MX.sym('x', 2)

def pendulum_ode(x):
    theta = x[0]
    omega = x[1]
    dtheta = omega
    domega = - (g / l) * ca.sin(theta)
    return ca.vertcat(dtheta, domega)

# Create CasADi function for the ODE
f = ca.Function('f', [x], [pendulum_ode(x)])

# Euler integration step
def euler_step(xk):
    return xk + dt * f(xk)

# RK4 integration step
def rk4_step(xk):
    k1 = f(xk)
    k2 = f(xk + dt/2 * k1)
    k3 = f(xk + dt/2 * k2)
    k4 = f(xk + dt * k3)
    return xk + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# Initialize arrays to store results
euler_results = np.zeros((N+1, 2))
rk4_results = np.zeros((N+1, 2))
time = np.linspace(0, T, N+1)

# Set initial conditions
euler_results[0, :] = [theta0, omega0]
rk4_results[0, :] = [theta0, omega0]

# Simulate using Euler and RK4 methods
x_euler = ca.DM([theta0, omega0])
x_rk4 = ca.DM([theta0, omega0])

for i in range(N):
    x_euler = euler_step(x_euler)
    x_rk4 = rk4_step(x_rk4)
    euler_results[i+1, :] = x_euler.full().flatten()
    rk4_results[i+1, :] = x_rk4.full().flatten()

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(time, euler_results[:, 0], label='Euler Method', linestyle='--')
plt.plot(time, rk4_results[:, 0], label='RK4 Method', linestyle='-')
plt.title('Pendulum Simulation: Euler vs RK4')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid(True)
plt.show()