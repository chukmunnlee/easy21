import sys
import pickle as pickle
from env import Easy21

import numpy as np
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

easy21 = Easy21()

EPISODE = 5

episodes = pickle.load(open(sys.argv[1], 'rb'))

# ((dealer, player), reward)
print('Episodes: %d' %len(episodes))
EPISODE = len(episodes)

reward = lambda s: s[1]
value = lambda s, val: val[s] if s in val else 0
count = lambda s, cnt: cnt[s] if s in cnt else 1

val = {}
cnt = {}

for i, ep in enumerate(episodes[:EPISODE]):
   curr = {}
   for j, step in enumerate(ep):
      st = step[0]
      if st in curr:
         continue
      curr[st] = sum(map(reward, ep[j + 1:]))

   for j, st in enumerate(curr):
      val[st] = curr[st] + value(st, val)
      cnt[st] = count(st, cnt) + 1

value_pi = [ (r[0], r[1]/cnt[r[0]]) for r in val.items() ]
x = [ r[0][0] for r in value_pi ] #dealer 0 - 10
y = [ r[0][1] for r in value_pi ] #player 0 - 21
z = [ r[1] for r in value_pi ]

print('x = ', x, len(x))
print('y = ', y, len(y))
print('z = ', z, len(z))

xi = np.linspace(0, 10, 20)
yi = np.linspace(0, 21, 42)
print('xi = ', xi, len(xi))
print('yi = ', yi, len(yi))

zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')


X, Y = np.meshgrid(xi, yi)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X = dealer')
ax.set_ylabel('Y = player')
ax.set_zlabel('Z = REWARD')

#https://stackoverflow.com/questions/18923502/plotting-contour-and-wireframe-plots-with-matplotlib
ax.plot_wireframe(X, Y, zi, rstride=1, cstride=1)

plt.title('Easy21 for %d episodes' %EPISODE)

plt.show()
