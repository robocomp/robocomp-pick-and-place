import gym, sys, time, math
from gym import spaces
import spaces as spcs
import numpy as np
from numpy import linalg as LA
sys.path.append('/home/robocomp/software/CoppeliaSim_Edu_V4_3_0_Ubuntu20_04/programming/zmqRemoteApi/clients/python')
from zmqRemoteApi import RemoteAPIClient

class EnvKinova_gym(gym.Env):

    #################################
    ## -- GYM INTERFACE METHODS -- ##
    #################################
    def __init__(self, n_actions):
        super(EnvKinova_gym, self).__init__()
        print('Loading environment')
        
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
        self.action_space = spcs.set_action_space(n_actions)
        # print("-------ACTION SPACE", self.action_space)
        action = self.action_space.sample()
        # print("-------ACTION", action)
        observation, _, done, _ = self.step(action)
        assert not done
        self.observation_space, self.n = spcs.set_observation_space(observation)
        self.goal = [0, 0]

    def step(self, action, iter=0):
        print("step() --> ACTION:", action)
        sim_act = [int(action[0]), int(action[1]), 0, 0, 0]
        
        if self.__interpretate_action(sim_act):
            self.sim.callScriptFunction("do_step@gen3", 1, sim_act)
        else:
            # print("INCORRECT ACTION: values not in [-1, 0, 1]")
            return None

        observation = self.__observate()
        # print("OBSERVATION IN STEP", observation)
        exit, reward, arrival, far, dist = self.__reward_and_or_exit(observation, iter)
        self.current_step += 1
        
        info = {
            "arrival": arrival,
            "far": far,
            "dist": dist,
        }

        return observation, reward, exit, info

    def reset(self):
        # print("RESET", "STEP:", self.current_step)
        # self.goal = self.sim.callScriptFunction("reset@gen3", 1) 

        self.close()
        time.sleep(0.1)
        self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)
        self.sim.loadScene("/home/robocomp/robocomp/components/robocomp-pick-and-place/etc/kinova_rl.ttt")
        self.sim.startSimulation()
        time.sleep(0.1)

        self.current_step = 0
        obs = self.__observate()
        return obs

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
        # return obs
        # return {"dist_x": obs["dist_x"], "dist_y": obs["dist_y"]}
        return np.array([obs["dist_x"], obs["dist_y"]])


    def __reward_and_or_exit(self, observation, iter):
        # return exit, reward, arrival, far, dist
        dist = np.linalg.norm(observation[:2])

        if dist > 0.1:
            return True, -10, 0, 1, dist
        
        elif dist < 0.005:
            return True, 100 - iter, 1, 0, dist

        else:
            rwrd = (0.1 - dist) * 100
            return False, int(rwrd), 0, 0, dist
    