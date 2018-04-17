import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 11)
y = np.sin(x)

xnew = np.linspace(0, 10, 100);
legend = []
legend_text = [ 'sin' ]

plt.plot(x, y)

for k in ['nearest', 'zero', 'linear', 'quadratic', 'cubic']:
   f = interpolate.interp1d(x, y, kind=k)
   legend.append(plt.plot(xnew, f(xnew), label=k))
   legend_text.append(k)

plt.legend(legend_text)
plt.show()
