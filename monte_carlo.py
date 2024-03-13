import random as rd
import matplotlib.pyplot as plt
import pygame

class MonteCarloControl():
    def __init__(self, env, epsilon, gamma, num_episodes):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.q_table = {}       # {(state):[q1,q2,q3,q4]}   
        self.visited_count = {} # {((state), action):count}
        self.policy  = {}       # {(state):policy}      
        self.reached_goal = 0
        self.reached_hole = 0
        self.reward_sum = 0
        self.reward_lists = []
        self.episode_lists = []
        self.episode_lengths = []

    def init_q_table(self):
        num_actions = 4
        map_size = len(self.env.map)
        for x in range(map_size):
            for y in range(map_size):
                if (x,y) in self.env.get_holes() or (x,y) == self.env.get_goal():
                    self.q_table[(x, y)] = [0] * num_actions
                else:
                    self.q_table[(x,y)] = [rd.random(), rd.random(), rd.random(), rd.random()]
    
    def init_policy(self):
        map_size = len(self.env.map)
        for x in range(map_size):
            for y in range(map_size):
                self.policy[(x,y)] = rd.randint(0,3) #random int between 0 and 3 inclusivw

    def init_agent(self):
        self.init_q_table()
        self.init_policy()
    
    def generate_episode(self, i):
        # initial state and (s,a,r)
        observation = self.env.reset()
        state_action_reward = []
        terminated = False
        num_of_steps = 0
        print(i) #for debugging

        while num_of_steps < 5000 and not terminated:
            policy = self.epsilon_greedy(observation)
            next_observation, reward, terminated = self.env.step(policy)
            state_action_reward.append((observation, policy, reward))
            #print(observation) #for debugging
            
            if terminated:
                if reward == 1:
                    self.reached_goal += 1
                if reward == -1:
                    self.reached_hole += 1

                self.reward_lists.append((self.reward_sum + reward)/(i+1))
                self.episode_lists.append(i)
                self.episode_lengths.append(len(state_action_reward)+1)
                self.reward_sum += reward
                break
            #update current state
            observation = next_observation
            num_of_steps += 1
        return state_action_reward

    def calculate_g(self, state_action_reward):
        steps = len(state_action_reward)
        G = 0
        g_list = [] #initialize g-table

        # loop from terminal state
        for i in range(steps-1, -1, -1):
            s = state_action_reward[i][0]
            a = state_action_reward[i][1]
            r = state_action_reward[i][2]

            G = r + self.gamma * G
            g_list.append((s, a, G))
        
        return reversed(g_list)
    
    def evaluate_policy(self, g_list):
        #print("Before: ", self.q_table)
        taken = set()
        for state_action in g_list:
            #print(state_action)
            state      = state_action[0]
            action_idx = state_action[1]
            g_value    = state_action[2]

            if (state, action_idx) not in self.visited_count:
                self.visited_count[(state, action_idx)] = 1
            else:
                self.visited_count[(state, action_idx)] += 1

            visited = self.visited_count[(state, action_idx)]          
            #update the q-value with the average
            if state_action not in taken:  #take first value only for first visit
                self.q_table[state][action_idx] = ((visited-1)/visited) * self.q_table[state][action_idx] + (1/ visited) * g_value
                taken.add(state_action)

    def update_policy(self):
        for state in self.policy:
            self.policy[state] = self.greedy(state) # give action with max-q      
    
    def epsilon_greedy(self, state):
        greedy = self.greedy(state)
        prob = rd.uniform(0,1)
        if prob <= self.epsilon:
            return rd.randint(0, 3)
        else:
            return greedy
    
    def greedy(self, state):
        return self.q_table[state].index(max(self.q_table[state]))

    def display(self):
        # Plot success vs failed attempt
        fig, ax = plt.subplots()
        label = ["Success Attempt", "Failed Attempt"]
        result = [self.reached_goal, self.reached_hole]
        bar_labels = ["blue", "red"]
        bar_colors = ["tab:blue", "tab:red"]

        ax.bar(label, result, label=bar_labels, color=bar_colors)

        ax.set_ylabel("Number of attempts")
        ax.set_xlabel("Success vs Failed Attempt")
        ax.legend(title="Legend")

        plt.show()

        # Plot average reward vs episodes
        plt.ylabel("Average reward")
        plt.xlabel("Number of episodes")
        plt.title("Average reward vs Number of episodes")
        plt.plot(self.episode_lists, self.reward_lists)
        plt.show()

        # Plot episode length vs episodes
        plt.ylabel("Episode length")
        plt.xlabel("Number of episodes")
        plt.title("Episode length vs Number of episodes")
        plt.plot(self.episode_lists, self.episode_lengths)
        plt.show()
    
    def visual(self):
        map_size = len(self.env.map)
        black = (0, 0, 0)
        white = (255, 255, 255)
        cell_size = 80
        side_length = map_size * cell_size
        status = True

        pygame.init()
        screen = pygame.display.set_mode((side_length, side_length))
        #clock = pygame.time.Clock()
        screen.fill(white)

        while status:
            for x in range(0, side_length, cell_size):
                for y in range(0, side_length, cell_size):
                    #setup grid
                    rect = pygame.Rect(x, y, cell_size, cell_size)
                    pygame.draw.rect(screen, black, rect, 1)

                    #draw all icons
                    x_transform = int(x/cell_size)
                    y_transform = int(y/cell_size)

                    if self.env.map[y_transform][x_transform] == 'H':
                        hole_img = pygame.image.load("/home/ooimax/me5406/p1/pothole.jpg")
                        hole_img = pygame.transform.scale_by(hole_img, cell_size/hole_img.get_width())
                        hole_icon = hole_img.convert()
                        screen.blit(hole_icon, (x,y))

                    elif self.env.map[y_transform][x_transform] == 'G':
                        frisbee_img = pygame.image.load("/home/ooimax/me5406/p1/frisbee.jpg")
                        frisbee_img = pygame.transform.scale_by(frisbee_img, cell_size/frisbee_img.get_width())
                        frisbee_icon = frisbee_img.convert()
                        screen.blit(frisbee_icon, (x,y))     

                    else: # draw policy for the rest
                        arrow_img = pygame.image.load("/home/ooimax/me5406/p1/arrow.jpg")
                        arrow_img = pygame.transform.scale_by(arrow_img, cell_size/arrow_img.get_width())
                        if self.policy[(x_transform, y_transform)] == 1: #down
                            arrow_img = pygame.transform.rotate(arrow_img, 90)
                        if self.policy[(x_transform, y_transform)] == 2: #right
                            arrow_img = pygame.transform.rotate(arrow_img, 180)
                        if self.policy[(x_transform, y_transform)] == 3: #up
                            arrow_img = pygame.transform.rotate(arrow_img, 270)  
                        arrow_icon = arrow_img.convert()
                        screen.blit(arrow_icon, (x,y))          

            pygame.display.update()

            for i in pygame.event.get():
                if i.type == pygame.QUIT:
                    status = False
        pygame.quit()
    
    def run(self):
        self.init_agent()

        for i in range(self.num_episodes):
            state_action_reward = self.generate_episode(i)
            g_table = self.calculate_g(state_action_reward)
            self.evaluate_policy(g_table)
            self.update_policy()
        
        self.display()
        self.visual()

        return self.q_table, self.policy, self.reached_goal, self.reached_hole


            
