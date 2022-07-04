import numpy as np
from EnvKinova_gym import EnvKinova_gym
import utilities as U
import time
import gym

from stable_baselines3 import DQN, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

env = EnvKinova_gym()
# env = gym.make('CartPole-v1')

print(env.observation_space)
print(env.action_space)
# print(3)
model = SAC('MlpPolicy', env, seed=0, learning_starts=1000, tensorboard_log="/home/vamsianumula/Desktop/logs/")
# print(1)
checkpoint_callback = CheckpointCallback(save_freq=2000, save_path='/home/vamsianumula/Desktop/logs/checkpoints/', name_prefix='check')

model.learn(total_timesteps=int(1e3),callback=checkpoint_callback)
# print(2)
model.save("final")

# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
# print(mean_reward)