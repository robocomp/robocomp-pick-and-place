import gym, sys, time, math
from gym import spaces
import numpy as np
from numpy import linalg as LA
sys.path.append('/home/robocomp/software/CoppeliaSim_Edu_V4_3_0_rev10_Ubuntu20_04/programming/zmqRemoteApi/clients/python')
from zmqRemoteApi import RemoteAPIClient

class EnvKinova_gym(gym.GoalEnv):

    #################################
    ## -- GYM INTERFACE METHODS -- ##
    #################################
    def __init__(self):
        super(EnvKinova_gym, self).__init__()
        print('Program started')
        
        # API CLIENT
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')

        # VARS
        self.possible_values = [-1, 0, 1]
        self.max_steps = 200
        self.current_step = 0

        # SCENE
        self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)
        self.sim.loadScene("/home/robocomp/robocomp/components/robocomp-pick-and-place/etc/kinova_rl.ttt")
        self.sim.startSimulation()
        time.sleep(1)

        # SPACES
        self.action_space = spaces.Box(low=-1*np.ones((5,)),high=np.ones((5,)),dtype=np.float64)
        self.observation_space = spaces.Dict({
            "observation":spaces.Box(low=-50*np.ones((17,)),high=50*np.ones((17,)),dtype=np.float64),
            "achieved_goal": spaces.Box(low=-50*np.ones((2,)),high=50*np.ones((2,)),dtype=np.float64),
            "desired_goal":spaces.Box(low=-50*np.ones((2,)),high=50*np.ones((2,)),dtype=np.float64)
        })

        self.dist = 0
        self.n=100^2
        self.goal= np.array([0.3,-0.589])

    def step(self, action):
        sim_act = self.__get_action(action)
        
        # if self.__interpretate_action(sim_act):
        if True:
            self.sim.callScriptFunction("do_step@gen3", 1, sim_act)
        else:
            print("INCORRECT ACTION: values not in [-1, 0, 1]")
            return None

        observation = self.__observate()
        reward = self.compute_reward(observation['achieved_goal'], observation['desired_goal'],None)
        done = self._terminate()

        self.current_step += 1
        info = {}
        return observation, reward, done, info

    def compute_reward(self,achieved_goal, desired_goal, info):
        dist = np.sqrt(np.square(achieved_goal-desired_goal).sum())
        self.dist=dist
        # if dist>0.05:
        #     reward=-1000
        # elif dist < 0.005:
        #     reward=1000
        # else:
        #     reward= (1 - self.__normalize(dist, 0, 2)) * 10
        if dist<0.05:
            reward = 1
        elif dist<0.2:
            reward = 0
        else:
            reward = -1
        
        return reward

    def reset(self):
        _ = self.sim.callScriptFunction("reset@gen3", 1) 
        # self.goal=np.array(self.goal[0:2])
        # print("Goal:",self.goal)
        self.current_step = 0
        obs = self.__observate()
        # print(obs["observation"][0:2])
        return obs

    def close(self):
        self.sim.stopSimulation()
        self.sim.setInt32Param(self.sim.intparam_idle_fps, self.defaultIdleFps)
        print('Program ended')

    def _terminate(self):
        if self.dist<0.05 or self.current_step>=self.max_steps:
            return True
        return False

    def __get_action(self, action):
        # print("ac:",action)
        ac = action.tolist()
        ac[4]=int(ac[4])
        return ac

    def __interpretate_action(self, action):
        return all(list(map(lambda x: x in self.possible_values, action)))

    def __observate(self):
        obs = self.sim.callScriptFunction("get_observation@gen3", 1) 
        # print("obs:")
        
        obs = self.__process_obs(obs)
        # print(obs[0:2])
        observ={}
        observ["observation"] = obs
        observ["desired_goal"] = self.goal
        observ["achieved_goal"] = np.array([obs[0],obs[1]])
        return observ

    def __process_obs(self,obs):
        state = []
        state+= obs["pos"][0]
        state+= [obs["dist_x"],obs["dist_y"],obs["dist_z"]]
        state+= obs["gripL"][1]
        state+= obs["gripR"][1]
        state+= [obs["gripper"]]
        # print("state:", state)
        # print("len:",len(state))
        return np.array(state)


    def __normalize(self, x, min_val, max_val):
        return (x - min_val) / (max_val + min_val)
