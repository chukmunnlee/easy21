import sys
import pickle

episodes = pickle.load(open(sys.argv[1], 'rb'))

print('episodes = ', episodes)
print('len = ', len(episodes))
