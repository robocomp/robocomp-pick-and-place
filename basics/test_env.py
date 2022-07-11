import numpy as np
from EnvKinova_gym import EnvKinova_gym
import utilities as U
import time

env = EnvKinova_gym()

# CONSTANTS
N_EPISODES = 10
MAX_ITER_EPISODE = 100

rewards = []

for i in range(N_EPISODES):
    s = env.reset()
    done = False  
    rew_ep = 0

    for j in range(MAX_ITER_EPISODE):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)

        rew_ep += reward
        if done:
            break
        s = next_state

    print(f"Eps: {i}, Reward: {rew_ep}")
    rewards.append(rew_ep)

    if i!=0 and i%9==0:
        avg = np.mean(rewards[-10:-1])
        print(f"Average reward for 10 episodes: {avg}") 