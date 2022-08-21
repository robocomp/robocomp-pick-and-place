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
        self.sim.loadScene("/home/robocomp/robocomp/components/robocomp-pick-and-place/etc/kinova_rl_grasp.ttt")
        self.sim.startSimulation()
        time.sleep(1)

        # SPACES
        self.action_space = spaces.Box(low=-1*np.ones((5,)),high=np.ones((5,)),dtype=np.float64)
        self.observation_space = spaces.Dict({
            "observation":spaces.Box(low=-200*np.ones((23,)),high=200*np.ones((23,)),dtype=np.float64),
            "achieved_goal": spaces.Box(low=-200*np.ones((1,)),high=200*np.ones((1,)),dtype=np.float64),
            "desired_goal":spaces.Box(low=-200*np.ones((1,)),high=200*np.ones((1,)),dtype=np.float64)
        })

        self.dist = 0
        self.n=100^2
        self.init_h=0

    def step(self, action):
        sim_act = self.__get_action(action)
        self.sim.callScriptFunction("do_step@gen3", 1, sim_act)

        observation = self.__observate()
        obs = observation["observation"]

        # gr_l=np.array(obs[10:13])
        # gr_r=np.array(obs[13:16])
        # gL= np.sqrt(gr_l.dot(gr_l))
        # gR = np.sqrt(gr_r.dot(gr_r))
        
        reward = self.compute_reward(observation['achieved_goal'], observation['desired_goal'],None)
        done = self._terminate()

        self.current_step += 1
        info = {}
        return observation, reward, done, info

    def compute_reward(self,achieved_goal, desired_goal, info):
        print("ag:", type(achieved_goal))
        print("dg:",desired_goal)

        self.dist = abs(achieved_goal-desired_goal)
        print("dist:",self.dist)

        # delta_h = abs(achieved_goal-desired_goal)
        # print("del_h:",delta_h)

        if self.dist<0.005:
            return 0
        reward = -1
        return reward

    def reset(self):
        goal = self.sim.callScriptFunction("reset@gen3", 1) 
        self.init_h=goal[2]
        
        self.current_step = 0
        obs = self.__observate()
        return obs

    def close(self):
        self.sim.stopSimulation()
        self.sim.setInt32Param(self.sim.intparam_idle_fps, self.defaultIdleFps)
        print('Program ended')

    def _terminate(self):
        if self.dist<0.005 or self.current_step>=self.max_steps:
            return True
        return False

    def __get_action(self, action):
        ac = action.tolist()
        ac[4]=int(round(ac[4]))
        return ac

    def __interpretate_action(self, action):
        return all(list(map(lambda x: x in self.possible_values, action)))

    def __observate(self):
        obs = self.sim.callScriptFunction("get_observation@gen3", 1) 
        obs = self.__process_obs(obs)

        observ={}
        observ["observation"] = obs
        observ["desired_goal"] = self.init_h+0.5
        observ["achieved_goal"] = obs[2]
        return observ

    def __process_obs(self,obs):
        state = []
        state+= obs["pos"][0]
        state+= [obs["dist_x"],obs["dist_y"],obs["dist_z"]]
        state+= obs["gripL"][1]
        state+= obs["gripR"][1]
        state+= [obs["gripper"]]
        state+= obs["fingerL"][1]
        state+= obs["fingerR"][1]
        return np.array(state)

    def __normalize(self, x, min_val, max_val):
        return (x - min_val) / (max_val + min_val)
