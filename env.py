import time
import sys
import numpy as np
import pickle as pickle

PLAYER_SKIP_VALUE = 18
DEALER_SKIP_VALUE = 17

class Easy21:

   HIT = 0
   STICK = 1

   def __init__(self):
      np.random.seed(round(time.time()))

   def draw(self):
      val = round(np.random.uniform(1, 11, None))
      return val if val > 0 else 1

   def is_red(self):
      return -1 if (round(np.random.uniform(0, 3)) % 3) == 1 else 1

   def is_bust(self, val):
      return (val < 1) or (val > 21)

   #(dealer, player sum), action
   def step(self, state, action):
      dealer = state[0]
      player = state[1]

      if (self.HIT == action):

         if self.is_bust(player):
            return (state, -1, True)

         card = self.is_red() * self.draw()
         new_state = (dealer, player + card)
         bust = self.is_bust(new_state[1])
         reward = -1 if bust else 0

         return (new_state, reward, bust)

      #STICK
      if self.is_bust(dealer):
            return (state, 1, True)

      card = self.is_red() * self.draw()
      dealer = dealer + card
      if self.is_bust(dealer): 
         return ((dealer, player), 1, True)

      if (dealer < DEALER_SKIP_VALUE):
         return ((dealer, player), 0, False)

      if (dealer == player):
         return ((dealer, player), 0, True)

      if (dealer > player):
         return ((dealer, player), -1, True)

      if (dealer < player):
         return ((dealer, player), 1, True)

if __name__ == '__main__':

   EPISODE = 20 if len(sys.argv) <= 1 else int(sys.argv[1])

   easy21 = Easy21()

   episodes = []

   for i in range(EPISODE):

      print('=== Episode %d ===' %i)

      per_episode = []
      state = (easy21.draw(), easy21.draw())

      #player's move
      while True:
         new_state, reward, bust = easy21.step(state, easy21.HIT)
         state = new_state

         per_episode.append((state, reward))

         if bust or new_state[1] >= PLAYER_SKIP_VALUE:
            break

      if not bust:
         while True:
            new_state, reward, bust = easy21.step(state, easy21.STICK)

            state = new_state

            per_episode.append((state, reward))

            if bust:
               break

      episodes.append(per_episode)

   pickle.dump(episodes, open('episodes_%d_%d.pickle' %(EPISODE, round(time.time())), 'wb'))
