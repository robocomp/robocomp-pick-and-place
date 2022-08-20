import gym, sys, time, math
from gym import spaces
import numpy as np
from numpy import linalg as LA
sys.path.append('/home/robocomp/software/CoppeliaSim_Edu_V4_3_0_rev10_Ubuntu20_04/programming/zmqRemoteApi/clients/python')
from zmqRemoteApi import RemoteAPIClient

class EnvKinova_gym(gym.Env):

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
        self.max_steps = 200
        self.current_step = 0

        # SCENE
        self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)
        self.sim.loadScene("/home/robocomp/robocomp/components/robocomp-pick-and-place/etc/kinova_rl_grasp.ttt")
        self.sim.startSimulation()
        time.sleep(1)

        # SPACES
        self.action_space = spaces.Box(low=-1*np.ones((5,)),high=np.ones((5,)),dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf*np.ones((26,)),high=np.inf*np.ones((26,)),dtype=np.float32)

        self.init_h=0
        self.goal_h = 0


    def step(self, action):
        sim_act = self.__get_action(action)
        self.sim.callScriptFunction("do_step@gen3", 1, sim_act)

        obs = self.__observate()

        # print(obs)
        gr_l=np.array(obs[20:23])
        gr_r=np.array(obs[23:26])
        gL= np.sqrt(gr_l.dot(gr_l))
        gR = np.sqrt(gr_r.dot(gr_r))
        
        print("gL:",gL,"gR:",gR)
        # print("gR:",gR)

        done = False

        if self.current_step>=self.max_steps:
            done = True
        
        if (gL>10 or gR>10):
            reward=-1000
            done=True
            self.close()
            self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
            self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)
            self.sim.loadScene("/home/robocomp/robocomp/components/robocomp-pick-and-place/etc/kinova_rl_grasp.ttt")
            self.sim.startSimulation()
            time.sleep(1.5)
        elif self.grasp_check(gL,gR,action) and abs(obs[2]-self.goal_h)<0.05:
            reward = 10
            done=True
        elif self.grasp_check(gL,gR,action) and (obs[2]>self.init_h):
            reward = 1
        else:
            reward = -0.025

        self.current_step += 1
        info = {}
        return obs, float(reward), done, info

    def grasp_check(self, gL,gR, action):
        if gL>3 and gR>3 and action[4]==-1:
            print('Grasp detected')
            return True
        return False

    def reset(self):
        _ = self.sim.callScriptFunction("reset@gen3", 1)
        self.current_step = 0
        obs = self.__observate()
        self.init_h = obs[2]
        self.goal_h = self.init_h+0.3
        # print(f'Initial height:{self.init_h}, Goal height:{self.goal_h}')
        return obs

    def close(self):
        self.sim.stopSimulation()
        self.sim.setInt32Param(self.sim.intparam_idle_fps, self.defaultIdleFps)
        print('Program ended')

    def __get_action(self, action):
        ac = action.tolist()
        ac[4]=int(round(ac[4]))
        # print(ac)
        return ac

    def __observate(self):
        obs = self.sim.callScriptFunction("get_observation@gen3", 1) 
        observ = self.__process_obs(obs)
        return observ

    def __process_obs(self,obs):
        state = []
        state+= obs["block"][0] #7
        state+= obs["pos"][0][0:3] #3
        state+= [obs["dist_x"],obs["dist_y"],obs["dist_z"]] #3
        state+= obs["gripL"][1] #3
        state+= obs["gripR"][1] #3
        state+= [obs["gripper"]] #1
        state+= obs["fingerL"][1] #3
        state+= obs["fingerR"][1] #3
        return np.array(state,dtype=np.float32)
