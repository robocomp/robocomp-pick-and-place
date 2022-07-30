from EnvKinova_gym import EnvKinova_gym
from stable_baselines3 import DQN, HerReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

env = EnvKinova_gym()
goal_selection_strategy = 'future'

model = DQN(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
        online_sampling=True,
        max_episode_length=200,
    ),
    verbose=1,
)

print(env.observation_space)
print(env.action_space)

checkpoint_callback = CheckpointCallback(save_freq=2000, save_path='/home/vamsianumula/Desktop/logs/HER/checkpoints/', name_prefix='HER')
model.learn(total_timesteps=int(1e5),eval_freq=1000,callback=checkpoint_callback)
model.save("final")