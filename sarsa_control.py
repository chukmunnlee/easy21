import sys
import time
from tqdm import tqdm
import pickle as pickle
import numpy as np
import matplotlib.pyplot as plt

from env import Easy21

class Policy:

   def __init__(self, q_value, stats, N0):
      self.N0 = N0
      self.q_value = q_value
      self.stats = stats
      np.random.seed(round(time.time()))

   def epsilon_greedy(self, st):
      #act = self.q_value.action_values(st)
      act = self.q_value[st]
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

   def state_action_pair(self):
      return self.state_action.keys()

   def state_action_count(self, st, act):
      return self.state_action[(st, act)] if st in self.state else 0

class Eligibility:

   def __init__(self):
      self.e = {}

   def update(self, hyp_gamma, hyp_lambda):
      scale = hyp_lambda * hyp_gamma
      for i in self.e:
         self.e[i] = scale * self.e[i]

   def __getitem__(self, key):
      return self.e[key] if key in self.e else 0

   def __setitem__(self, key, value):
      self.e[key] = value

class QValue:

   def __init__(self):
      self.q = {}

   def __getitem__(self, key):
      st = key
      return self.q[st] if st in self.q else [0 , 0]

   def __setitem__(self, key, value):
      self.q[key] = value

   def action_values(self, st, act = None):
      if st not in self.q:
         self.q[st] = [0, 0]

      if not act:
         return self.q[st]

      return self.q[st][act]

   def update(self, st, act):
      self.action_values(st)
      self.q[st] = act

   def exportQValues(self):
      opt_qvals = {}
      for st in self.q:
         opt_qvals[st] = np.array(self.q[st])
      return opt_qvals

   def dump(self):
      print('q_value = ', len(self.q.keys()))

if __name__ == '__main__':

   pfile = './episodes/mc_qstar_10m.pickle' if len(sys.argv) <= 1 else sys.argv[1]

   mc_qval = pickle.load(open(pfile, 'rb'))

   qval_optimal = {}

   for st in mc_qval:
      qval_optimal[st] = np.array(mc_qval[st])

   N0 = 100
   EPISODES = 1000
   hyp_gamma = 1
   easy21 = Easy21()

   mse = []

   ep_count = np.arange(1, 1001, dtype=np.int8)

   for hyp_lambda in np.arange(0, 1.1, 0.1):

      stats = Stats()
      q_value = QValue()
      policy = Policy(q_value, stats, N0)
      eligibility = Eligibility()

      mse_per_ep = []

      for i in tqdm(range(EPISODES), desc='lambda = %.1f' %hyp_lambda):

         terminate = False

         #(dealer, player)
         episode = []
         st = (easy21.draw(), easy21.draw())
         act = policy.epsilon_greedy(st)
         while not terminate:
            next_st, r, terminate = easy21.step(st, act)
            next_act = policy.epsilon_greedy(next_st)

            episode.append((st, act, r, next_st))

            stats.update(st, act)

            qval = q_value[st]
            next_qval = q_value[next_st]

            alpha = 1 / stats.state_action_count(st, act)
            delta = r + (hyp_gamma * next_qval[next_act]) - qval[act]
            ad = alpha * delta
            eligibility[st, act] += 1

            for a_st, a_act in stats.state_action_pair():
               a_act_val = q_value[a_st]
               a_act_val[a_act] += ad * eligibility[(a_st, a_act)]
               q_value[a_st] = a_act_val

            #update all traces
            eligibility.update(hyp_gamma, hyp_lambda)

            st, act = next_st, next_act

         mse_per_ep.append(\
               np.sum([ np.square(qval_optimal[st] - np.array(q_value[st])) for st in qval_optimal ]) / (len(qval_optimal) * 2))

      mse.append(mse_per_ep)

   mse = np.array(mse)

   lam = 0.1
   for i in mse[1: -1]:
      plt.plot(i, label='lambda=%.1f' %lam)
      lam += lam

   plt.xlabel('Episode')
   plt.ylabel('MSE')

   plt.show()

