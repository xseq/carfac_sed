# CARFAC (cascaded asymmetric resonator with fast-acting compression)
from pylab import *
from scipy import signal


n_sec  = 100 
x_low  = 0.1                         
x_high = 0.9                         
x = linspace(x_high, x_low, n_sec)

f = 165.4 * (10**(2.1 * x) - 1)
a0 = cos(2 * pi * f / fs)
c0 = sin(2 * pi * f / fs)
damping = 0.2

r = 1 - damping * 2 * pi * f / fs
h = c0
g = (1 - 2 * a0 * r + r * r) / (1 - (2 * a0 - h * c0) * r + r * r)  


