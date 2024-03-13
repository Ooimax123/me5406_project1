import environment
import monte_carlo as mc
import q_learning as ql
import sarsa as sar

# "4x4" for 4x4 map, "10x10" for 10x10 map
env = environment.FrozenLake("4x4")

# Hyperparameters for tuning
epsilon = 0.1
gamma = 0.9
alpha = 0.1
num_episodes = 10000

observation, _ = env.reset()

mc_agent = mc.MonteCarloControl(env, epsilon, gamma, num_episodes)
Q_table, final_policy, reached_goal, reached_hole = mc_agent.run()

#q_agent = ql.QLearning(env, alpha, epsilon, gamma, num_episodes)
#Q_table, final_policy, reached_goal, reached_hole = q_agent.run()

#sarsa_agent = sar.SARSA(env, alpha, epsilon, gamma, num_episodes)
#Q_table, final_policy, reached_goal, reached_hole = sarsa_agent.run()

print("Q_table: ", Q_table)
print("Policy: ", final_policy)
#print(return_per_episode)
print("Num of goal reached: ", reached_goal)
print("Num of hole reached: ", reached_hole)

