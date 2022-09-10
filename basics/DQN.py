import math
import random
import matplotlib
import matplotlib.pyplot as plt
from itertools import count

from EnvKinova import *
from dqn_aux import *
import data_from_dims as DFD
import q_aux as QA

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
EPOCHS = 25000

##########################################################
N_DIMS = 4                                               #
                                                         #
# Number of dimensions considered in the environment.    #
# All the training process and the environment structure #
# will depend onthis constant.                           #
##########################################################

OBS_SIZE, N_ACTIONS, ACTION_FUNC, REWARD_FUNC = DFD.get_data(N_DIMS)
# print("OBS", OBS_SIZE, "POS ACTIONS", N_ACTIONS)
env = EnvKinova(OBS_SIZE, N_DIMS, REWARD_FUNC, ACTION_FUNC)

policy_net = DQN(OBS_SIZE, N_ACTIONS, device).to(device)
target_net = DQN(OBS_SIZE, N_ACTIONS, device).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            act = policy_net(state).argmax().view(1, 1)
            # print("net's action", act)
            return act
    else:
        act = torch.tensor([[random.randrange(N_ACTIONS)]], device=device, dtype=torch.long)
        # print("random action", act)
        return act


episode_durations = []
acc_rewards = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy(), 'bo')
    
    if len(durations_t) >= 100:  # 100 episode durations
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), 'b')

    rewards_t = list(map( lambda x:x/10, acc_rewards))
    rewards_t = torch.tensor(rewards_t, dtype=torch.float)
    plt.plot(rewards_t.numpy(), 'ro')

    if len(rewards_t) >= 100:  # 100 episode rewards
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), 'r')

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize_model():
    # print("Optimizing...")
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    non_final_next_states = torch.cat(batch.next_state)
    non_final_next_states = torch.reshape(non_final_next_states, (128,OBS_SIZE))
    state_batch = torch.cat(batch.state)
    state_batch = torch.reshape(state_batch, (128,OBS_SIZE))
    action_batch = torch.cat(batch.action)
    action_batch = action_batch.type(torch.int64)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values = target_net(non_final_next_states).max(1)[0].detach()
    
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

for i_episode in range(EPOCHS):
    # Initialize the environment and state
    state = env.reset()
    for t in count():
        action = select_action(state)

        # print("DQN TAKEN ACTION:", action.item, "DQN ACTION", QA.index2action(int(action.item())))
        next_state, reward, done, info = env.step(QA.index2action(int(action.item()), N_DIMS), t)
        reward = torch.tensor([reward], device=device)

        # print("STATE", state, "NEXT STATE", next_state)
        memory.push(torch.Tensor(state), action, torch.Tensor(next_state), reward)
        state = next_state

        optimize_model()
        if done:
            episode_durations.append(t + 1)
            acc_rewards.append(info["acc_reward"])
            plot_durations()
            break
        
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('TRAINING ENDED :)')
env.close()
plt.ioff()
plt.show()
