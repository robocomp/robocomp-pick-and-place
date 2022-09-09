# GSoC'22 RoboComp project: Reinforcement Learning for pick and place operations

8th September 2022

## Introduction
The aim of this project is to tackle the Robotic Manipulation task using Deep Reinforcement Learning. We will train a Kinova Gen3 arm simulated in CoppeliaSIm, to grasp and place objects at a desired position. Robotic Manipulation involves high dimensional state space and pretty complex dynamics, making it a challenging task to solve. A custom OpenAI Gym environment is created in Python, for simulating the Kinova Gen3 arm. The MDP for this problem is formulated with a continuous state space, continuous/discrete action space. The agent will be trained using Model-Free Deep RL algorithms such as Soft Actor Critic(SAC) and advanced techniques like Hind Sight Expereince Replay (HER) can be augmented to allow sample-efficient learning for this complex environment. 

## Environment

### Objective
The aim of the project is to make a Open AI Gym wrapper for the exisiting robotic arm model in CoppeliaSim. The gym wrapper creation eases the process of training our agent. The currently available library implementations of state-of-the-art Deep RL algorithms require the custom environment to follow this gym.Env structure. A standard wrapper has been built until now.The environment supports both continuous and discrete action spaces.

### Environment Description

#### State Space

A 29 dimensional continuous state space is considered, comprising of:

|        Info                           |  Dimensions |
| -------------------------             |  ---|
| Block pose: 3 coords+ 4 quaternions  |  7  |
| Block velocity                        |  3  |
| Block angular velocity                 |  3  |
| Gripper tip position corods           |  3  |
| Relative position of block w.r.t tip  |  3  |
| Grip force sensors (left & right)     |  2  |
| Finger force sensors (left & right)   |  2  |
| Rel. position b/w left&right fingers   |  3  |
| Gripper velocity                      |  3  |

#### Action space

5 dimensional action space in either discrete or continuous setting.

|        Info               |  Discrete  |  Continuous |
| ------------------------- |  ---       |  ---   |
| Move arm in x-direction   |  {-1,0,1}  |[-1,1]  |
| Move arm in y-direction   |  {-1,0,1}  |[-1,1]  |
| Move arm in z-direction   |  {-1,0,1}  |[-1,1]  |
| Move wrist                |  {-1,0,1}  |[-1,1]  |
| Open/Close the gripper    |  {-1,0,1}  |[-1,1], but will berounded off to {-1,0,1}  |

#### Collision Detection

Collision Detection is a important aspect for the environment as it prevents arm to crash into block, table and such. The force data from the left and right finger sensors is used. The magintude of force sensors is obtained and if that exceeds a certain threshold, a collsion is detected. The threshold is finetuned from observations of various training episodes involving collisions.

#### Grasp Detection
Similar to collision detection, if the force magintudes obtained from the gripper sensors exceeds a certain fiinetuned threshold, a grasp is detected. In the training phase, this would be a very useful feature to have in the reward function, where a certain reward is achieved for a successful grasp.  

### Further steps

#### Goal Environment for goal-conditioning with HER

Since, the task of pick and place is quite complex, we want to use to leverage the idea of goal-conditioning. With goal-conditioning, each episode is considering as a success by treating the achieved terminal state as a virtual goal state. Hindsight Experience Replay(HER) is used to achieved the goal conditioning for our agent. In order to use HER, our environment need to be modified into a gym.goalEnv structure, where the observation space consists of state, achieved goal and desired goal, and the reward for each time step will be computed based on this structure. This goal env will be created and tested. 

## Training 

### Training Objective
The goal of the current phase is for the robot arm to reach the block, grasp it and lift it to a desired height above ground.

### Reward

The agent will get rewarded as follows:

|        State                           |  Reward | Terminal? |
| -------------------------             |  ---|   ----   |
| Arm is far way from the block         |  -100  |  Yes  |
| Collision detected                        |  -100  |  Yes  |
| Grasp detected and dh>0                 |  1000\*dh_norm | No |
| Goal height reached                 |  10,000 | Yes |

#### Notation
dh:= change in object height from ground \
dh_norm:= Normailzed dh

*\*The reward structure is subject to change*

### Algorithms

Soft Actor Critic (SAC) is chosen for training in continuous action space setting.

### Trained agent demo

*TODO*

### Reward curve

*TODO: Will be added once hyperparameter tuning is done.*

### Futher Steps
 - The next step would be train the arm to place the block at desired position after grasping, by modifying rewards.
 - Modify exisitng env to support Goal Conditioning and train the arm using SAC, along with HindSight Experience Replay(HER) replay buffer to achieve a more robust and sample efficient training of the agent.
