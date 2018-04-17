import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 11)
y = np.sin(x)

xnew = np.linspace(0, 10, 100);
legend = []
legend_text = [ 'sin' ]

plt_num = 321
fig = plt.figure()
ax = fig.add_subplot(plt_num)
ax.plot(x, y)
ax.legend(['sin'])

for k in ['nearest', 'zero', 'linear', 'quadratic', 'cubic']:
   plt_num = plt_num + 1
   ax = fig.add_subplot(plt_num)
   f = interpolate.interp1d(x, y, kind=k)
   ax.plot(x, y)
   ax.plot(xnew, f(xnew))
   ax.legend(['sin', k])

plt.show()
