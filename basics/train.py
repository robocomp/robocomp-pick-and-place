import numpy as np
from EnvKinova_gym import EnvKinova_gym
import utilities as U
import time

env = EnvKinova_gym()

# CONSTANTS
N_EPISODES = 10000
MAX_ITER_EPISODE = 100

rewards = []

for i in range(N_EPISODES):
    s = env.reset()
    s_i = U.state2index(s)
    done = False
    
    rew_ep = 0

    for j in range(MAX_ITER_EPISODE):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        
        action_i = U.action2index(action)
        next_state_i = U.state2index(next_state)
        
        # print("EPISODE", i, "ITER", j, "STATE", s_i, "ACTION", action_i)

        rew_ep += reward

        if done:
            break
        
        s = next_state
        s_i = next_state_i

    print(f"Eps: {i}, Reward: {rew_ep}")
    rewards.append(rew_ep)

    if i!=0 and i%30==0:
        avg = np.mean(rewards[-30:])
        print(f"Average reward for 30 episodes: {avg}") 

