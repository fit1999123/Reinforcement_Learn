import numpy as np
import gym
import random
import matplotlib.pyplot as plt


env = gym.make("CliffWalking-v0")
env = env.unwrapped
action_space_n = env.action_space.n

state_n = env.observation_space.n

episodes = 10000

alpha = 0.5

n_step = 20

gamma = 0.7

epsilon = 1.0

def arg_max(lst):
    return lst.index(max(lst))

class robot():

    def __init__(self):
        

        self.state = env.reset()

        self.lst = []

        self.Q_table = [[0 for i in range(action_space_n)] for j in range(state_n)]

 
    def Q_learning(self,epsilon):

            
        for i in range(n_step):
                    
            r = random.random()
                    
            if r>epsilon:
                        
                action = arg_max(self.Q_table[self.state])

            else:
                    
                action = int(env.action_space.sample())
                    
                    
            new_state,reward,terminated,truncated,info =  env.step(action)
            
            next_action = arg_max(self.Q_table[new_state])
                
            if terminated or truncated:
                    
                self.Q_table[self.state][action] += alpha*(reward-self.Q_table[self.state][action])
                        
                self.state = env.reset()
                    
                break
                    
            else:
                        
                self.Q_table[self.state][action] += alpha*(reward+gamma*(self.Q_table[new_state][next_action]-self.Q_table[self.state][action]))
                        
                self.state = new_state
                    

    def test(self,algorithm):

        global epsilon

        for i in range(episodes):
            
            if i >= 0.25*episodes and i< 0.5*episodes:
                
                epsilon = 0.5
            
            elif i>=0.5*episodes:
                
                epsilon = 0.1
            
            algorithm(epsilon)

            self.lst.append(sum(self.Q_table[5]))


my_robot = robot()

my_robot.test(my_robot.Q_learning)

lst = my_robot.lst

x = [i for i in range(episodes)]
            
fig = plt.figure()

ax1 = fig.add_subplot(1,1,1)
ax1.plot(x,lst)
plt.show()
plt.close()







env = gym.make("CliffWalking-v0",render_mode="human")
my_robot.state = env.reset()

while True:

    action = arg_max(my_robot.Q_table[my_robot.state])
    lst2 = env.step(action)
    my_robot.state = lst2[0]
    if lst2[2] == True or lst2[3] == True:
                
        break
    
print("Game Over")

