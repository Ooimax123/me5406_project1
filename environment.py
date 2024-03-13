import numpy as np

maps = {"4x4":  [['R', 'F', 'F', 'F'],
                 ['F', 'H', 'F', 'H'],
                 ['F', 'F', 'F', 'H'],
                 ['H', 'F', 'F', 'G']],
        "10x10": [['R', 'F', 'F', 'H', 'H', 'H', 'F', 'F', 'F', 'F'],
                  ['F', 'F', 'F', 'F', 'F', 'H', 'F', 'F', 'F', 'F'],
                  ['F', 'F', 'F', 'F', 'F', 'H', 'F', 'H', 'H', 'H'],
                  ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
                  ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
                  ['F', 'H', 'F', 'H', 'F', 'F', 'F', 'F', 'F', 'F'],
                  ['F', 'H', 'F', 'F', 'H', 'F', 'H', 'H', 'F', 'F'],
                  ['F', 'H', 'F', 'H', 'H', 'F', 'F', 'H', 'F', 'F'],
                  ['F', 'F', 'F', 'F', 'H', 'F', 'H', 'H', 'F', 'F'],
                  ['H', 'H', 'F', 'H', 'F', 'F', 'F', 'F', 'H', 'G']]}

class FrozenLake():

    def __init__(self, choice):
        self.map = maps[choice]
        self.size = len(self.map)**2 #asssume map is square in size
        self.window_size = 500       #for window display
        self.action_to_direction = {0:[-1,0], #left
                                    1:[0,1],  #down
                                    2:[1,0],  #right
                                    3:[0,-1]} #up
        self.state = (0,0) #initialize robot to be at (0,0)

    def get_holes(self):
        terminals = set()
        for row in range(len(self.map)):
            for col in range(len(self.map)):
                if self.map[row][col] == 'H':
                    terminals.add((col, row))
        return terminals
    
    def get_goal(self):
        return (len(self.map)-1, len(self.map)-1)

    def step(self, action):
        direction = self.action_to_direction[action]
        new_state = (self.state[0] + direction[0], self.state[1] + direction[1])

        # only update when it still within map
        if 0 <= new_state[0] < len(self.map) and 0 <= new_state[1] < len(self.map):
            self.state = new_state

        observation  = self.state 
        hole         = False
        goal         = False

        if observation in self.get_holes():
            hole = True
        if observation == self.get_goal():
            goal = True
            
        #assign reward
        if hole:
            reward = -1

        elif goal:
            reward = 1

        else:
            reward = 0
        
        terminated = goal or hole

        return observation, reward, terminated
    
    def reset(self):
        self.state = (0,0)
        return self.state
    

    
    