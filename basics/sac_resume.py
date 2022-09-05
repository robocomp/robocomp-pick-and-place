from sac_env import EnvKinova_gym
from stable_baselines3 import DQN, HerReplayBuffer,SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
import os

def get_idx(i):
    files= [f for f in os.listdir(".") if os.path.isfile(f) and i in f and "checkpoint" in f]
    file = max(files)
    idx = file[:-4].split('_')[-1]
    
    idx_next = int(idx)+1
    return i+"_"+idx, i+"_"+str(idx_next)

env = EnvKinova_gym()

i='0' # Training trail number

idx1,idx2 = get_idx(i)
# print(idx1, idx2)

print(f"Loading checkpoint{idx1}...")
model = SAC.load(f'checkpoint{idx1}',env=env)
model.load_replay_buffer(f"rb{idx1}")

try:
    model.learning_starts=0
    checkpoint_callback = CheckpointCallback(save_freq=2000, save_path=f'/home/vamsianumula/Desktop/logs/SAC_{idx2}/checkpoints/', name_prefix=f'SAC{idx2}')
    model.learn(total_timesteps=int(2e6),eval_freq=5000,callback=checkpoint_callback,reset_num_timesteps=False)
    # model.save(f"sac{idx2}")
    print(f"Saving checkpoint{idx2}...")
    model.save(f"checkpoint{idx2}")
    model.save_replay_buffer(f"rb{idx2}")
except Exception as e:
    print(f"Saving checkpoint{idx2}...")
    model.save(f"checkpoint{idx2}")
    model.save_replay_buffer(f"rb{idx2}")
except KeyboardInterrupt as e:
    print(f"Saving checkpoint{idx2}...")
    model.save(f"checkpoint{idx2}")
    model.save_replay_buffer(f"rb{idx2}")


