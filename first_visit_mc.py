import sys
import pickle as pickle
from env import Easy21

import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

easy21 = Easy21()

EPISODE = 20000

episodes = pickle.load(open(sys.argv[1], 'rb'))

# ((dealer, player), reward)
print('Episodes: %d' %len(episodes))

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

player = [] #X
dealer = [] #Y
reward = [] #Z
for r in value_pi:
   dealer.append(r[0][0])
   player.append(r[0][1])
   reward.append(r[1])

X, Y = np.meshgrid(player, dealer)
print('X = ', X)
print('Y = ', Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X = player')
ax.set_ylabel('Y = dealer')
ax.set_zlabel('Z = REWARD')

#https://stackoverflow.com/questions/18923502/plotting-contour-and-wireframe-plots-with-matplotlib
ax.plot_wireframe(X, Y, reward)

plt.show()
