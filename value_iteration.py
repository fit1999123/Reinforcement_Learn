import random 
import numpy as np
import matplotlib.pyplot as plt


gamma =0.9
max_iterations = 10000
theta = 0.0001
actions = [0,1] # left right
world =  [1,2,3,4,5,6,7,8]
reward = [0,2,1,-1,3,-3,-7,5]
right = [[0.3,0.7] for i in range(len(world))]
left = [[0.7,0.3] for i in range(len(world))]
balance = [[0.5,0.5] for i in range(len(world))]

policy = balance


for i in range(len(left)):
    
    if i == 3 or i == 6 or i ==7:
       
       left[i] = [0,0]
    elif i == 5:

       left[i] = [1,0]
       

for i in range(len(right)):
    
    if i == 3 or i == 6 or i ==7:
       
       right[i] = [0,0]
    elif i == 5:

       right[i] = [1,0]

for i in range(len(balance)):
    
    if i == 3 or i == 6 or i ==7:
       
       balance[i] = [0,0]
    elif i == 5:

       balance[i] = [1,0]
       


class robot:

    def __init__(self):

        self.q = [0 for i in range(len(world))]

        self.q_a = [[0,0] for i in range(len(world))]


    def next_step(self,action):

        if action==0:
            if self.loc == 1:
                return 2
            elif self.loc == 2:
                return 4
            elif self.loc == 3:
                return  5
            elif self.loc == 5:
                return 7
            elif self.loc == 6:
                return  8
            else:
                return self.loc
            
        elif action==1:
            if self.loc == 1:
                return 3
            elif self.loc == 2:
                return 5
            elif self.loc == 3:
                return 6
            elif self.loc == 5:
                return 8
            else:
                return self.loc
            
        
         
 
my_robot = robot() 




def value_estimation():
    
    for i in range(max_iterations):
        detla = 0
        for s in range(len(world)):
            
            q_s = [0 for i in range(len(actions))]
            
            my_robot.loc = 1
            
            for a in actions:
                
                q_s[a] = policy[s][a]*my_robot.q[world.index(my_robot.next_step(a))]
                        
                q_s[a] *= gamma

                q_s[a] += reward[s]
            
            best_value = max(q_s)
            
            detla = max(detla,abs(best_value-my_robot.q[s]))
            
            my_robot.q[s] = best_value
        
        # print(my_robot.q)    
            
        if detla<theta:
            break
        
def value_iter():
    
    value_estimation()

    for s in range(len(world)):
        q_s = [0 for i in range(len(actions))]
        
        my_robot.loc = world[s]
        
        for a in actions:
          
            q_s[a] = policy[s][a]*my_robot.q[world.index(my_robot.next_step(a))]
                        
            q_s[a] *= gamma

            q_s[a] += reward[s]
          
            
        best_action = np.argmax(np.array(q_s))
        policy[s] = np.eye(len(actions))[best_action].tolist()


    
    return policy

policy = value_iter()






    
    
lst = []
    
action = ["L","R","T"]
    
for i in policy:
            
    lst.append(action[i.index(max(i))])

for i in [3,6,7]:
           
    lst[i] = action[2]
     
     
print(lst)