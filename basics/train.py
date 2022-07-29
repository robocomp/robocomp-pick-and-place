from EnvKinova_gym import EnvKinova_gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env

env = EnvKinova_gym()

check_env(env, warn=True, skip_render_check=True)

print(env.observation_space)
print(env.action_space)

model = DQN('MlpPolicy', env, seed=0, verbose=1,learning_starts=1000, tensorboard_log="/home/vamsianumula/Desktop/logs/")
checkpoint_callback = CheckpointCallback(save_freq=2000, save_path='/home/vamsianumula/Desktop/logs/checkpoints/', name_prefix='check')
model.learn(total_timesteps=int(1e5),eval_freq=1000,callback=checkpoint_callback)
model.save("final")