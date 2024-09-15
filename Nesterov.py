from numpy import zeros, array, size, dot, random, linalg
import time

N = 200
M = 300

A = random.sample((M, N))

from numpy import sin, pi
x_model = array([sin(2*pi*i/(N-1)) for i in range(N)])

b = dot(A, x_model)

Delta = 0.5
random.seed(1)
b_delta = b + Delta*(random.random(len(b))-
                     random.random(len(b)))

delta2 = sum((b - b_delta)**2)/len(b)

x = zeros(N)

def Nesterov_acceleration_scheme(A, b_delta, x, alpha, delta2):
    s = 1
    x_prev = zeros(size(x))
    # z_0 = 0
    z = x_prev
    omega = 1 / linalg.norm(A)
    omega = omega ** 2
    #while s <= N ** 2 :
    while sum((dot(A,x) - b_delta)**2)/len(b_delta) >= delta2 :
        x_prev = x
        x = z + omega * dot(A.T, b_delta - dot(A, z))
        z = x + (s - 1) / (s + alpha) * (x - x_prev)
        s = s + 1
    return x, s

time_start = time.time()

alpha = 1
x, s = Nesterov_acceleration_scheme(A, b, x, alpha, delta2)

total_time = time.time() - time_start

from matplotlib.pyplot import figure, axes, show
from numpy import arange
fig = figure()
ax = axes(xlim=(0, N), ylim=(-1.5, 1.5))
ax.set_xlabel('i'); ax.set_ylabel('x[i]')
ax.plot(arange(N), x_model, '-g', lw=7)
ax.plot(arange(N), x, '-r', lw=2)
show()

print(f'Time for consecutive algorithm: {total_time:9.3f} sec')
