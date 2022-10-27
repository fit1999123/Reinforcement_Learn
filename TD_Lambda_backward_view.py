import random
import numpy as np
import matplotlib.pyplot as plt

gamma = 0.7
n_iterations = 10000
n_step = 4
actions = [0,1]  ##left right
world =  [ 1, 2, 3, 4, 5, 6, 7, 8]
reward = [ 0, 2, 1,-1, 3,-3,-7, 5]
alpha = 0.01
Lambda = 0.0
greedy = 0.5

class robot:

    def __init__(self,pos):

        self.pos = pos
    
        self.q = [[0,0] for i in range(8)]

        self.e = [[0,0] for i in range(8)]

    def move(self,action):

        if action == 0:

            if self.pos == world[0]:

                pos = world[1]

            elif self.pos == world[1]:

                pos = world[3]

            elif self.pos == world[2]:

                pos =world[4]

            elif self.pos == world[4]:

                pos = world[6]

            elif self.pos == world[5]:

                pos = world[7]

        elif action == 1:

            if self.pos == world[0]:

                pos = world[2]

            elif self.pos == world[1]:

                pos = world[4]

            elif self.pos == world[2]:

                pos = world[5]

            elif self.pos == world[4]:

                pos = world[7]

        
        return pos

def argmax(lst):

    return lst.index(max(lst))

my_robot = robot(1)

def TD(Lambda):


    for i in range(n_step):

        

        if my_robot.pos == 4 or my_robot.pos == 7 or my_robot.pos == 8:

            break


        else:

            if my_robot.pos == 6:

                a = 0

            else:
                
                r = random.random()

                if r>=greedy:
                    
                    a = argmax(my_robot.q[world.index(my_robot.pos)])
                else:

                    a = random.choice(actions)

            new_state = my_robot.move(a)

            get_reward = reward[world.index(new_state)]

            delta = get_reward + gamma*argmax(my_robot.q[world.index(new_state)]) - my_robot.q[world.index(my_robot.pos)][a]
      
            my_robot.e[world.index(my_robot.pos)][a] += 1

            for j in range(len(my_robot.q)):

                my_robot.q[j][a] += alpha*delta*my_robot.e[j][a]

                my_robot.e[j][a] *= gamma*Lambda

            my_robot.pos = new_state

lst = []
         
for i in range(n_iterations):

    TD(Lambda)
    
    my_robot.pos = random.randint(1,8)

    lst.append(my_robot.q[4][1])

x = [i for i in range(len(lst))]

fig = plt.figure()

ax1 = fig.add_subplot(1,1,1)

ax1.plot(x,lst,color = "red",label= "TD(Lambda)")
plt.legend()

plt.show()
plt.close()

   
