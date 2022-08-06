from EnvKinova_gym import EnvKinova_gym
from stable_baselines3 import DQN, HerReplayBuffer,SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

env = EnvKinova_gym()
goal_selection_strategy = 'future'

model = SAC(
    "MultiInputPolicy",
    env,
    learning_starts=2000,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
        online_sampling=True,
        max_episode_length=200,
    ),
    verbose=1,
    seed=0,
    tensorboard_log="/home/vamsianumula/Desktop/logs/HER_SAC/"
)

print(env.observation_space)
print(env.action_space)

checkpoint_callback = CheckpointCallback(save_freq=2000, save_path='/home/vamsianumula/Desktop/logs/HER_SAC/checkpoints/', name_prefix='HER_SAC')
model.learn(total_timesteps=int(2e6),eval_freq=1000,callback=checkpoint_callback)
model.save("final")