from sac_env import EnvKinova_gym
from stable_baselines3 import DQN, HerReplayBuffer,SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

env = EnvKinova_gym()
goal_selection_strategy = 'future'

i='3_0'  #change this when starting new training

model = SAC("MlpPolicy",
    env,
    use_sde=True,
    learning_starts=10000,
    # policy_kwargs=dict(net_arch=[400, 300]),
    learning_rate=0.001,
    verbose=1,
    seed=0,
    tensorboard_log=f"/home/vamsianumula/Desktop/logs/SAC_{i}"
)

# model = SAC.load('/home/vamsianumula/Desktop/logs/SAC_71/checkpoints/SAC71_68000_steps.zip',env=env)

try:
    checkpoint_callback = CheckpointCallback(save_freq=2000, save_path=f'/home/vamsianumula/Desktop/logs/SAC_{i}/checkpoints/', name_prefix=f'SAC{i}')
    model.learn(total_timesteps=int(2e6),eval_freq=5000,callback=checkpoint_callback,reset_num_timesteps=False)
    model.save(f"checkpoint{i}")
    model.save_replay_buffer(f"rb{i}")
except Exception as e:
    model.save(f"checkpoint{i}")
    model.save_replay_buffer(f"rb{i}")
except KeyboardInterrupt as e:
    model.save(f"checkpoint{i}")
    model.save_replay_buffer(f"rb{i}")

# print(env.observation_space)
# print(env.action_space)