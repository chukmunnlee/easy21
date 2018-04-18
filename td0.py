import sys
import pickle as pickle

import numpy as np
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

#hyper parameters
ALPHA=0.001 #step size
GAMMA = 1 #discounting
INITIAL_ESTIMATE = 0.02

episodes = pickle.load(open(sys.argv[1], 'rb'))

#EPISODES = 10
EPISODES = len(episodes)
print('Episodes: %d' %EPISODES)
print('\tfile: %s' %sys.argv[1])

def value(s, st):
   if s in st:
      return st[s]
   return INITIAL_ESTIMATE

states = {}
for ep in episodes[:EPISODES]:
   curr = ep[0]
   for e in ep[1:]:
      curr_st = curr[0]
      curr_rew = e[1]
      next_st = e[0]
      states[curr_st] = value(curr_st, states) + \
            (ALPHA * (curr_rew + (GAMMA * value(next_st, states)) - value(curr_st, states)))
      curr = e

x = [ r[0][0] for r in states.items() ] # dealer
y = [ r[0][1] for r in states.items() ] # player
z = [ r[1] for r in states.items() ]

xi = np.linspace(0, 10, 20)
yi = np.linspace(0, 21, 42)
zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')

X, Y = np.meshgrid(xi, yi)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('X = dealer')
ax.set_ylabel('Y = player')
ax.set_zlabel('Z = REWARD')

X, Y = np.meshgrid(xi, yi)

#https://stackoverflow.com/questions/18923502/plotting-contour-and-wireframe-plots-with-matplotlib
ax.plot_wireframe(X, Y, zi, rstride=1, cstride=1)

plt.title('Easy21 using TD(0) for %d episodes' %EPISODES)

plt.show()
