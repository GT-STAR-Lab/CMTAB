import time
from debris_rss23 import debris_task
from fire_rss23 import fire_task
from search_rss23 import search_task
import numpy as np

Q= np.array([[ 0.1, 4, 1, 0.18],
        [ 0.1, 1, 5, 0.12],
        [0.1, 1, 2, 0.54],
        [0.2, 2, 2, 0.24]])
# Order of traits: Speed, water, payload, sensing radius

X = np.array([[3,0,0,1],
     [1,3,1,1],
     [0,1,3,1]])
# Order of tasks: douse fire, move debris, search area

#print(X.shape, Q.shape)
visualize = True

reward = fire_task(X, Q, visualize)
print("Fire: ", reward)

reward = debris_task(X, Q, visualize)
print("Debris: ", reward)

reward = search_task(X, Q, visualize)
print("Search area: ", reward)
