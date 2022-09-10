import gym, sys, time
import numpy as np
from numpy import linalg as LA
sys.path.append('/home/robocomp/software/CoppeliaSim_Edu_V4_3_0_Ubuntu20_04/programming/zmqRemoteApi/clients/python')
from zmqRemoteApi import RemoteAPIClient

import spaces as spcs
import data_from_dims as DFD

class EnvKinova(gym.Env):

    #################################
    ## -- GYM INTERFACE METHODS -- ##
    #################################
    def __init__(self, obs_size, n_dims, reward_func, action_func):
        super(EnvKinova, self).__init__()
        # print('Loading environment')

        self.obs_size = obs_size
        self.action_func = action_func
        self.reward_func = reward_func
        
        # API CLIENT
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')

        # VARS
        self.possible_values = [-1, 0, 1]
        self.max_steps = 200
        self.current_step = 0
        self.acc_reward = 0
        self.p_2d_dist = 0
        self.p_3d_dist = 0

        # SCENE
        self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)
        self.sim.loadScene("/home/robocomp/robocomp/components/robocomp-pick-and-place/etc/kinova_rl.ttt")
        self.sim.startSimulation()
        time.sleep(1)

        # SPACES
        self.action_space = spcs.set_action_space(n_dims)
        # print("-------ACTION SPACE", self.action_space)
        action = self.action_space.sample()
        # print("-------ACTION", action)
        observation, _, done, _ = self.step(action)
        assert not done
        self.observation_space, self.n = spcs.set_observation_space(observation)

    def step(self, action, iter=0):
        sim_act = self.action_func(action)
        print("EnvKinova.step() --> ACTION:", action)
        
        if self.__interpretate_action(sim_act):
            self.sim.callScriptFunction("do_step@gen3", 1, sim_act)
        else:
            # print("INCORRECT ACTION: values not in [-1, 0, 1]")
            return None

        observation = self.__observate()
        # print("OBSERVATION IN STEP", observation)
        exit, reward, arrival, self.p_2d_dist, self.p_3d_dist = self.reward_func(observation, iter, self.p_2d_dist, self.p_3d_dist)
        self.current_step += 1
        self.acc_reward += reward
        
        info = {
            "arrival": arrival,
            "acc_reward": self.acc_reward,
        }

        return observation, reward, exit, info

    def reset(self):
        self.close()
        time.sleep(0.1)
        self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)
        self.sim.loadScene("/home/robocomp/robocomp/components/robocomp-pick-and-place/etc/kinova_rl.ttt")
        self.sim.startSimulation()
        time.sleep(0.1)

        self.current_step = 0
        self.acc_reward = 0
        self.previous_dist = 0
        obs = self.__observate()
        return obs

    def close(self):
        self.sim.stopSimulation()
        self.sim.setInt32Param(self.sim.intparam_idle_fps, self.defaultIdleFps)

    ####################################
    ## -- PRIVATE AUXILIAR METHODS -- ##
    ####################################

    def __interpretate_action(self, action):
        return all(list(map(lambda x: x in self.possible_values, action)))

    def __observate(self):
        obs = {"pos": [[0, 0, 0]]}
        obs = self.sim.callScriptFunction("get_observation@gen3", 1) 
        array_obs = np.array([obs["dist_x"], obs["dist_y"], obs["dist_z"],
               LA.norm(obs["fingerL"]), LA.norm(obs["fingerR"]),
               LA.norm(obs["gripL"]), LA.norm(obs["gripR"])])

        return array_obs[:self.obs_size]
    