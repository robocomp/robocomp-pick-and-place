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
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Dict({
            "observation":spaces.Box(low=-50*np.ones((2,)),high=50*np.ones((2,)),dtype=np.float64),
            "achieved_goal": spaces.Box(low=-50*np.ones((2,)),high=50*np.ones((2,)),dtype=np.float64),
            "desired_goal":spaces.Box(low=-50*np.ones((2,)),high=50*np.ones((2,)),dtype=np.float64)
        })
        self.n=100^2

    def step(self, action):
        ac1, ac2 = self.__get_action(action)
        sim_act = [int(ac1), int(ac2), 0, 0, 0]
        
        if self.__interpretate_action(sim_act):
            self.sim.callScriptFunction("do_step@gen3", 1, sim_act)
        else:
            print("INCORRECT ACTION: values not in [-1, 0, 1]")
            return None

        observation = self.__observate()

        exit, reward, _  = self.__reward_and_or_exit()
        self.current_step += 1
        
        info = {}

        return observation, reward, exit, info

    def compute_reward(self,achieved_goal, desired_goal, info):
        
        dist = math.sqrt((achieved_goal[0]-desired_goal[0])**2 + (achieved_goal[1]-desired_goal[1])**2)

        if dist < 0.005:
            reward=0
        else:
            reward=-1

        return reward

    def reset(self):
        #print("RESET", "STEP:", self.current_step)
        self.goal = self.sim.callScriptFunction("reset@gen3", 1) 
        self.observation_space["desired_goal"] = self.goal

        self.current_step = 0
        obs = self.__observate()

        self.observation_space["desired_goal"] = self.goal
        self.observation_space["achieved_goal"] = obs
    
        return 

    def close(self):
        self.sim.stopSimulation()
        self.sim.setInt32Param(self.sim.intparam_idle_fps, self.defaultIdleFps)
        print('Program ended')

    ####################################
    ## -- PRIVATE AUXILIAR METHODS -- ##
    ####################################

    def __interpretate_action(self, action):
        return all(list(map(lambda x: x in self.possible_values, action)))

    def __observate(self):
        obs = {"pos": [[0, 0, 0]]}
        obs = self.sim.callScriptFunction("get_observation@gen3", 1) 
        # return {"distX":obs["dist_x"], "distY":obs["dist_y"]}
        return np.array([obs["dist_x"], obs["dist_y"]])

    def __reward_and_or_exit(self):
        exit, reward, arrival, far = False, 0, 0, 0

        achieved_goal = self.observation_space["achieved_goal"]
        desired_goal = self.observation_space["desired_goal"]

        dist = math.sqrt((achieved_goal[0]-desired_goal[0])**2 + (achieved_goal[1]-desired_goal[1])**2)
        reward =self.compute_reward(achieved_goal, desired_goal, None)
        if dist > 0.1 or dist <0.005:
            exit = True

        return exit, reward, None

        ''' SIMPIFIED VERSION
        Goes away:           True, -1
        Reaches the target:  True,  1
        Else:                False, 0 '''
    
    

    def __normalize(self, x, min_val, max_val):
        return (x - min_val) / (max_val + min_val)

    def __get_action(self, action):
        x = action // 3
        y = action % 3
        return int(x-1), int(y-1)   