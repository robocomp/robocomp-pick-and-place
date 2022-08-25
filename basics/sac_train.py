from sac_env import EnvKinova_gym
from stable_baselines3 import DQN, HerReplayBuffer,SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

env = EnvKinova_gym()
goal_selection_strategy = 'future'

i='8'
model = SAC("MlpPolicy",
    env,
    use_sde=True,
    learning_starts=6000,
    verbose=1,
    seed=0,
    tensorboard_log=f"/home/vamsianumula/Desktop/logs/SAC_{i}"
)

# model = SAC.load('/home/vamsianumula/Desktop/logs/SAC_71/checkpoints/SAC71_68000_steps.zip',env=env)
# model = SAC.load('checkpoint7',env=env)
# model.load_replay_buffer("rb7")

try:
    # model.learning_starts=0
    checkpoint_callback = CheckpointCallback(save_freq=2000, save_path=f'/home/vamsianumula/Desktop/logs/SAC_{i}/checkpoints/', name_prefix=f'SAC{i}')
    model.learn(total_timesteps=int(2e6),eval_freq=5000,callback=checkpoint_callback)
    model.save(f"sac{i}")
except Exception as e:
    model.save(f"checkpoint{i}")
    model.save_replay_buffer(f"rb{i}")

# print(env.observation_space)
# print(env.action_space)