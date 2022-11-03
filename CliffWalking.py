import numpy as np
import gym
import random
import math
import time
import matplotlib.pyplot as plt


env = gym.make("CliffWalking-v0")
env = env.unwrapped
action_space_n = env.action_space.n

state_n = env.observation_space.n

episodes = 10000

alpha = 0.5

n_step = 20

gamma = 0.7

epsilon = 0.5

state,_ = env.reset()


Q_table = [[0 for i in range(action_space_n)] for j in range(state_n)]

def arg_max(lst):
    return lst.index(max(lst))





def Q_learning(epsilon):
    
    global state
    
    for i in range(n_step):
            
        # r = random.random()
            
        # if r>epsilon:
                
        #     action = arg_max(Q_table[state])

        # else:
            
        action = int(env.action_space.sample())
            
            
        new_state,reward,terminated,truncated,info =  env.step(action)

        next_action = arg_max(Q_table[new_state])
          
        if terminated or truncated:
            
            Q_table[state][action] += alpha*(reward-Q_table[state][action])
                
            state,_ = env.reset()
            
            break
            
        else:
                
            Q_table[state][action] += alpha*(reward+gamma*(Q_table[new_state][next_action]-Q_table[state][action]))
                
            state = new_state
             


lst = []




for i in range(episodes):
    
    if i >= 0.25*episodes and i< 0.5*episodes:
        
        epsilon = 0.5
    
    elif i>=0.5*episodes:
        
        epsilon = 0.1
    
    Q_learning(epsilon)

    lst.append(sum(Q_table[5]))







x = [i for i in range(episodes)]
    
fig = plt.figure()

ax1 = fig.add_subplot(1,1,1)
ax1.plot(x,lst)
plt.show()
plt.close()







env = gym.make("CliffWalking-v0",render_mode="human")
state,_ = env.reset()

while True:

    action = arg_max(Q_table[state])
    lst2 = env.step(action)
    state = lst2[0]
    if lst2[2] == True or lst2[3] == True:
        
        break
    
print("Game Over")

print(Q_table)