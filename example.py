import casadi as cas
import numpy as np
import math
import matplotlib.pyplot as plt

opti = casadi.Opti()

p = opti.variable(2,N)
x = p(1,:)
y = p(2,:)

V = 0.5*D*sum((x(1:N-1)-x(2:N)).^2+(y(1:N-1)-y(2:N)).^2)
V = V + g*sum(m*y)

opti.minimize(V)

opti.subject_to(p(:,1)==[-2;1])
opti.subject_to(p(:,end)==[2;1])

opti.solver('ipopt')
sol = opti.solve()

plot(sol.value(x),sol.value(y),'-o')
