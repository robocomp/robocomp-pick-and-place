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
        self.observation_space = spaces.Box(low=-np.inf*np.ones((29,)),high=np.inf*np.ones((29,)),dtype=np.float32)

        self.init_h=0
        self.goal_h = 0


    def step(self, action):
        sim_act = self.__get_action(action)
        self.sim.callScriptFunction("do_step@gen3", 1, sim_act)
        
        if sim_act[4]!=0:
            time.sleep(0.4)  
        obs = self.__observate()

        fL, fR = obs[21], obs[22]        
        done = False
        reward = 0

        if self.current_step>=self.max_steps:
            done = True
        
        if self.dist_check(obs):
            print('Arm is very far')
            done=True
            reward=-100
        elif obs[2]-self.init_h<0:
            done=True
            reward = -100
            print('Object is getting crushed')
        elif (fL>5 or fR>5):
            print('Collison detected')
            self.close()
            self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
            self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)
            self.sim.loadScene("/home/robocomp/robocomp/components/robocomp-pick-and-place/etc/kinova_rl_grasp.ttt")
            self.sim.startSimulation()
            time.sleep(4)
            reward=-100
            done=True
        elif self.grasp_check(obs,sim_act) and abs(obs[2]-self.goal_h)<0.05:
            print('Goal reached')
            reward = 1000
            done=True
        elif self.grasp_check(obs,sim_act):
            diff = obs[2]-self.init_h
            reward = 1
            print(f'Grasp detected with obj height diff: {diff}')
            if diff>0:
                reward += 10*self.__normalize(diff,0,self.goal_h-self.init_h)

        reward -= -0.1

        self.current_step += 1
        info = {}
        return obs, float(reward), done, info

    def grasp_check(self, obs, action):
        gL= obs[19]
        gR = obs[20]
        
        # print("gL:",gL,"gR:",gR, "ac:",action[4])

        if gL>0.15 and gR>0.15 and action[4]==-1:
            return True
        return False

    def dist_check(self,obs):
        dist = np.linalg.norm(obs[16:19])
        # print(dist)
        if dist<0.1:
            return False
        return True
    
    def reset(self):
        _ = self.sim.callScriptFunction("reset@gen3", 1)
        self.current_step = 0
        obs = self.__observate()
        self.init_h = obs[2]
        self.goal_h = self.init_h+0.1
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
        # print(obs)
        observ = self.__process_obs(obs)
        return observ

    def __process_obs(self,obs):
        state = []
        #object pose
        state+= obs["block"][0] #7

        #object velocity
        state+= obs["obj_vel"][0] #3
        #object ang velocity
        state+= obs["obj_vel"][1] #3

        state+= obs["pos"][0][0:3] #3
        state+= [obs["dist_x"],obs["dist_y"],obs["dist_z"]] #3

        gr_l= np.array(obs["gripL"][1]) 
        gr_r= np.array(obs["gripR"][1]) 
        gL= np.sqrt(gr_l.dot(gr_l)).item()
        gR = np.sqrt(gr_r.dot(gr_r)).item()
        fr_l= np.array(obs["fingerL"][1]) 
        fr_r = np.array(obs["fingerR"][1]) 
        fL= np.sqrt(fr_l.dot(fr_l)).item()
        fR = np.sqrt(fr_r.dot(fr_r)).item()

        state+=[gL,gR,fL,fR]#4 20,21,22,23

        f = np.array(obs["fL"][0][0:3]) - np.array(obs["fR"][0][0:3])
        state+= f.tolist() #3
        state+= obs["gripper_vel"][0] #3

        return np.array(state,dtype=np.float32)
    
    def __normalize(self, x, min_val, max_val):
        return (x - min_val) / (max_val + min_val)
