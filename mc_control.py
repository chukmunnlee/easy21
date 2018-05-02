import sys
import time
from tqdm import tqdm

import pickle as pickle
import numpy as np

from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

from env import Easy21

#action = lambda state, q_value: q_value[state] if state in q_value else (0, 0)
get_state_action = lambda state, q_value: q_value[state] if state in q_value else [0, 0]

# e = (dealer, player), action, reward, new_state
gain = lambda episode: sum([v[2] for v in episode])

def count(state, count_list):
   if state not in count_list:
      count_list[state] = 0
   count_list[state] = count_list[state] + 1

def epsilon(st, st_count):
   st_n = state_count[st] if st in state_count else 1
   return N0 / (N0 + st_n)

#epsilon greedy policy with decay
def policy(state, q_value):
   #if we don't know what to do, then HIT
   if state not in q_value:
      return Easy21.HIT

   a = get_state_action(state, q_value)
   #epislon decreases over time from more stochastic to more deterministic
   #correspond to more greedy
   if np.random.uniform(0, 1, None) <= epsilon(state, state_count):
      return a.index(max(a))

   #uniformly select any action from the the action space
   return a.index(min(a)) if np.random.uniform(0, 1, None) < 0.5 else a.index(max(a))

def generate_episode():
   # (dealer, player)
   state = (easy21.draw(), easy21.draw())
   episode = []

   while True:
      a = policy(state, q_value)

      new_state, reward, bust = easy21.step(state, a)

      episode.append((state, a, reward, new_state))

      if bust or (a == Easy21.STICK):
         return episode

      state = new_state

get_state_action = lambda state, q_value: q_value[state] if state in q_value else [0, 0]

#initialize random
np.random.seed(round(time.time()))

#tunables
EPISODE = 100 if len(sys.argv) <= 1 else int(sys.argv[1])
N0 = 100 * 2

state_count = {}
state_action_count = {}

# { (dealer, player_sum): (HIT, STICK), ... }
q_value = {}
lose = 0;
win = 0;

#start
easy21 = Easy21()

for i in tqdm(range(EPISODE)):
   episode = generate_episode()
   w = episode[-1][2] > 0
   #print('Episode: %d, len=%d, win=%s' %(i, len(episode), str(w)))
   #print('episode = ', episode)

   #for first visit
   visited = {}

   if w:
      win += 1
   else:
      lose += 1

   # e = (dealer, player), action, reward, new_state
   for idx, (from_st, a, rwd, new_st)  in enumerate(episode):
      count(from_st, state_count)
      count((from_st, a), state_action_count)

      pair_state_action = get_state_action(from_st, q_value)
      g = gain(episode[idx:])

      alpha = 1 / state_action_count[(from_st, a)]

      #print('alpha = ', alpha, ', before: state-action ', pair_state_action)
      pair_state_action[a] = pair_state_action[a] + (alpha * (g - pair_state_action[a]))
      #print('after: state-action ', pair_state_action)
      q_value[from_st] = pair_state_action

      #print('from: %s, action: %d, reward: %d, to: %s' %(str(from_st), a, rwd, str(new_st)))
      #print('pair state action ', pair_state_action)
      #print('e = ', episode[idx:], ', g = ', g)

#save the q_value
pickle.dump(q_value, open('episodes/q_value_%d_%d.pickle' %(EPISODE, round(time.time())), 'wb'))

dealer = [] # 0 - 10 X
player = [] # 0 - 21 Y
hit_qval = [] # -1 - 1 Z
stick_qval = [] # -1 - 1 Z
arg_max = [] # -1 - 1 Z
# { (dealer, player_sum): (HIT, STICK), ... }
for st in q_value:
   a = q_value[st]
   dealer.append(st[0])
   player.append(st[1])
   hit_qval.append(a[Easy21.HIT])
   stick_qval.append(a[Easy21.STICK])
   arg_max.append(max(a))

print('state count = ', len(state_count))
print('state action count = ', len(state_action_count))
print('win = %d, lose = %d' %(win, lose))

#print('dealer = ', dealer);
#print('player = ', player);
#print('hit_qval = ', hit_qval);
#print('stick_qval = ', stick_qval);

xi = np.linspace(0, 10, 22);
yi = np.linspace(0, 21, 42);
zi_hit = griddata((dealer, player), hit_qval, (xi[None, :], yi[:, None]), method='cubic')
zi_stick = griddata((dealer, player), stick_qval, (xi[None, :], yi[:, None]), method='cubic')
zi_argmax = griddata((dealer, player), arg_max, (xi[None, :], yi[:, None]), method='cubic')

X, Y = np.meshgrid(xi, yi)

fig = plt.figure()

ax = fig.add_subplot(221, projection='3d')
ax.set_xlabel('X = dealer')
ax.set_ylabel('Y = player')
ax.set_zlabel('Z = hit')
ax.set_title('HIT: episode: %d' %EPISODE)

ax.plot_wireframe(X, Y, zi_hit, rstride=1, cstride=1, color='blue')

ay = fig.add_subplot(222, projection='3d')
ay.set_xlabel('X = dealer')
ay.set_ylabel('Y = player')
ay.set_zlabel('Z = stick')
ay.set_title('STICK: episode: %d' %EPISODE)

ay.plot_wireframe(X, Y, zi_stick, rstride=1, cstride=1, color='red')

az = fig.add_subplot(223, projection='3d')
az.set_xlabel('X = dealer')
az.set_ylabel('Y = player')
az.set_zlabel('Z = argmax Q(s, a)')
az.set_title('V*(s): episode: %d' %EPISODE)

az.plot_wireframe(X, Y, zi_argmax, rstride=1, cstride=1, color='green')

plt.show()
