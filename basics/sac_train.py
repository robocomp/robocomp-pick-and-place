from sac_env import EnvKinova_gym
from stable_baselines3 import DQN, HerReplayBuffer,SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

env = EnvKinova_gym()
goal_selection_strategy = 'future'

# model = SAC("MlpPolicy",
#     env,
#     use_sde=True,
#     learning_starts=2000,
#     verbose=1,
#     seed=0,
#     tensorboard_log="/home/vamsianumula/Desktop/logs/SAC_1"
# )

model = SAC.load('/home/vamsianumula/Desktop/logs/SAC_1/checkpoints/SAC11_10000_steps.zip',env=env)
model.learning_starts=1000

print(env.observation_space)
print(env.action_space)

checkpoint_callback = CheckpointCallback(save_freq=2000, save_path='/home/vamsianumula/Desktop/logs/SAC_1/checkpoints/', name_prefix='SAC11')
model.learn(total_timesteps=int(2e6),eval_freq=1000,callback=checkpoint_callback)
model.save("final")