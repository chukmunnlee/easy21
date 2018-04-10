import sys
import pickle as pickle
from env import Easy21

easy21 = Easy21()

episodes = pickle.load(open(sys.argv[1], 'rb'))

print('Episodes: %d' %len(episodes))

for i, ep in enumerate(episodes):
   print(ep)
