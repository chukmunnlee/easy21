import sys
import pickle

from env import Easy21

argmax = lambda actions: actions.index(max(actions))

q_value = pickle.load(open(sys.argv[1], 'rb'))

print('q_value = %d' %len(q_value))

lose = 0
win = 0

def print_play(state):
   print('My card: %d, Dealer: %d' %(state[1], state[0]))

def print_action(action):
   if action == Easy21.HIT:
      print('\t===> HIT HIT <===')

   elif action == Easy21.STICK:
      print('\t===> STICK STICK <===')

while True:

   print('New Game')

   easy21 = Easy21()

   player = easy21.draw()
   dealer = easy21.draw()

   state = (dealer, player)

   while True:
      print_play(state)

      if state not in q_value:
         print('state = %s not in q_value' %str(state))
         sys.exit(-1)

      action = argmax(q_value[state])
      print('q_value = ', q_value[state], ': argmax = ', action)
      print_action(action)

      if action == Easy21.HIT:
         state, reward, bust = easy21.step(state, argmax(q_value[state]))
         print_play(state)
         if bust:
            print('Busted. You win')
            lose += 1

      else:
         break

   sys.exit(-1)

