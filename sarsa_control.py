import sys
import time
from tqdm import tqdm

import numpy as np

from env import Easy21

class Policy:

   def __init__(self, q_value, stats, N0):
      self.N0 = N0
      self.q_value = q_value
      self.stats = stats
      np.random.seed(round(time.time()))

   def epsilon_greedy(self, st):
      act = self.q_value.action_values(st)
      #act greedily
      if np.random.uniform(0, 1, None) > self.epsilon(st):
         return np.argmax(act)

      #act stochastic 
      return np.random.randint(Easy21.HIT, Easy21.STICK + 1)

   def epsilon(self, st):
      return N0 / (N0 + self.stats.state_count(st))

class Stats:

   def __init__(self):
      self.state = { }
      self.state_action = { }

   def update(self, st, act):
      if st not in self.state:
         self.state[st] = 0
      if (st, act) not in self.state_action:
         self.state_action[(st, act)] = 0

      self.state[st] += 1
      self.state_action[(st, act)] += 1

   def state_count(self, st):
      return self.state[st] if st in self.state else 0

   def state_action_count(self, st, act):
      return self.state_action[(st, act)] if st in self.state else 0

class QValue:

   def __init__(self):
      self.q = {}

   def action_values(self, st, act = None):
      if st not in self.q:
         self.q[st] = [0, 0]

      if not act:
         return self.q[st]

      return self.q[st][act]

   def update(self, st, act):
      self.action_values(st)
      self.q[st] = act

   def dump(self):
      print('q_values = ', self.q)

if __name__ == '__main__':
   N0 = 100
   EPISODES = 100
   gamma = 1
   stats = Stats()
   q_value = QValue()
   policy = Policy(q_value, stats, N0)
   easy21 = Easy21()

   for i in tqdm(range(EPISODES)):
      terminate = False

      #(dealer, player)
      episode = []
      st = (easy21.draw(), easy21.draw())
      act = policy.epsilon_greedy(st)
      while not terminate:
         next_st, r, terminate = easy21.step(st, act)
         next_act = policy.epsilon_greedy(next_st)

         stats.update(st, act)

         alpha = 1 / stats.state_action_count(st, act)
         qval = q_value.action_values(st)
         next_qval = q_value.action_values(next_st)

         qval[act] = qval[act] + (alpha * (r + (gamma * next_qval[next_act]) - qval[act]))
         q_value.update(st, qval)

         episode.append((st, act, r, next_st))
         print('>> ', episode)
         st, act = next_st, next_act

   q_value.dump()
