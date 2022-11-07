import numpy as np
import gym
import random
import matplotlib.pyplot as plt


env = gym.make("CliffWalking-v0")
env = env.unwrapped
action_space_n = env.action_space.n

state_n = env.observation_space.n

episodes = 20000

alpha = 0.8

n_step = 50

gamma = 0.7

epsilon = 1.0

def arg_max(lst):
    return lst.index(max(lst))

def greedy(Q_table,state,epsilon):
    
    r = random.random()
   
    
    if r>epsilon:
                        
        action = arg_max(Q_table[state])

    else:
                    
        action = int(env.action_space.sample())

    return action
    

    
                    

class robot():

    def __init__(self):
        

        self.state = env.reset()

        self.lst = []

        self.Q_table = [[0 for i in range(action_space_n)] for j in range(state_n)]

 
    def Q_learning(self,epsilon):

            
        for i in range(n_step):
                    
         
            action = greedy(self.Q_table,self.state,epsilon)


            new_state,reward,terminated,truncated,info =  env.step(action)
            
            next_action = arg_max(self.Q_table[new_state])
                
            if terminated or truncated:
                    
                self.Q_table[self.state][action] += alpha*(reward-self.Q_table[self.state][action])
                        
                self.state = env.reset()
                    
                break
                    
            else:
                        
                self.Q_table[self.state][action] += alpha*(reward+gamma*(self.Q_table[new_state][next_action]-self.Q_table[self.state][action]))
                        
                self.state = new_state
                    
    def SARSA(self,epsilon):

        for i in range(n_step):
                    
         
            action = greedy(self.Q_table,self.state,epsilon)


            new_state,reward,terminated,truncated,info =  env.step(action)
            
            next_action = greedy(self.Q_table,new_state,epsilon)
                
            if terminated or truncated:
                    
                self.Q_table[self.state][action] += alpha*(reward-self.Q_table[self.state][action])
                        
                self.state = env.reset()
                    
                break
                    
            else:
                        
                self.Q_table[self.state][action] += alpha*(reward+gamma*(self.Q_table[new_state][next_action]-self.Q_table[self.state][action]))
                        
                self.state = new_state

    def Expected_SARSA(self,epsilon):

        for i in range(n_step):
                    
         
            action = greedy(self.Q_table,self.state,epsilon)


            new_state,reward,terminated,truncated,info =  env.step(action)

            expected_value = sum(self.Q_table[new_state])/len(self.Q_table[new_state])

            if terminated or truncated:
                    
                self.Q_table[self.state][action] += alpha*(reward-self.Q_table[self.state][action])
                        
                self.state = env.reset()
                    
                break
                    
            else:
                        
                self.Q_table[self.state][action] += alpha*(reward+gamma*expected_value-self.Q_table[self.state][action])
                        
                self.state = new_state


    def test(self,algorithm):

        global epsilon

        for i in range(episodes):
            
            if i >= 0.25*episodes and i< 0.5*episodes:
                
                epsilon = 0.5
            
            elif i>=0.5*episodes:
                
                epsilon = 0.1
            
            algorithm(epsilon)

            self.lst.append(sum(self.Q_table[7]))

my_robot = robot()
my_robot2 = robot()
my_robot3 = robot()
my_robot.test(my_robot.Q_learning)
my_robot2.test(my_robot2.SARSA)
my_robot3.test(my_robot3.Expected_SARSA)
env = gym.make("CliffWalking-v0",render_mode="human")
my_robot.state = env.reset()

while True:

    action = arg_max(my_robot.Q_table[my_robot.state])
    lst2 = env.step(action)
    my_robot.state = lst2[0]
    if lst2[2] == True or lst2[3] == True:
                
        break
my_robot2.state = env.reset()
while True:

    action = arg_max(my_robot2.Q_table[my_robot2.state])
    result = env.step(action)
    my_robot2.state = result[0]
    if result[2] == True or result[3] == True:
                
        break
my_robot3.state = env.reset()
while True:

    action = arg_max(my_robot3.Q_table[my_robot3.state])
    result = env.step(action)
    my_robot3.state = result[0]
    if result[2] == True or result[3] == True:
                
        break




lst = my_robot.lst
lst2 = my_robot2.lst
lst3 = my_robot3.lst
x = [i for i in range(episodes)]
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x,lst)
ax.plot(x,lst2)
ax.plot(x,lst3)
plt.show()
plt.close()



