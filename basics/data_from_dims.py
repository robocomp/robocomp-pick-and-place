import numpy.linalg as LA

def get_data(dims):
    print("CONFIGURATION FOR", dims, "DIMENSIONS")
    if dims == 2:
        return 2, 3**dims, __action_2d, __reward_2d
    if dims == 3:
        return 5, 3**dims, __action_3d, __reward_3d
    if dims == 4:
        return 7, 3**dims, __action_4d, __reward_4d

#################################
##     -- 2  DIMENSIONS --     ##
#################################

def __action_2d(action):
    return [int(action[0]), int(action[1]), 0, 0, 0]

def __reward_2d(observation, iter, p_2d_dist, p_3d_dist):
    dist_2d = LA.norm(observation[:2])

    if dist_2d > 0.08:
        print("Far in 2D")
        return True, -5, 0, dist_2d, None
    
    elif dist_2d < 0.005:
        print("DONE :)")
        return True, 500 - iter, 1, dist_2d, None

    elif iter > 200:
        print("Too slow")
        return True, -20, 0, dist_2d, None

    else:
        rwrd =  10 if dist_2d < p_2d_dist else -3
        return False, rwrd, 0, dist_2d, None

#################################
##     -- 3  DIMENSIONS --     ##
#################################

def __action_3d(action):
    return [int(action[0]), int(action[1]), int(action[2]), 0, 0]

def __reward_3d(observation, iter, p_2d_dist, p_3d_dist):
    dist_2d = LA.norm(observation[:2])
    dist_3d = LA.norm(observation[:3])

    if observation[3] > 0.1 or observation[4] > 0.1:
        print("Collision")
        return True, 10 - (dist_3d*1000), 0, dist_2d, dist_3d

    elif dist_3d > 0.11:
        print("Far in 3D")
        return True, -10, 0, dist_2d, dist_3d

    elif dist_2d > 0.08:
        print("Far in 2D")
        return True, -30, 0, dist_2d, dist_3d
    
    elif dist_2d < 0.005:
        if dist_3d <= 0.01:
            print("DONE :)")
            return True, 500 - iter, 1, dist_2d, dist_3d
        else:
            print("Above the cube")
            return False, 20, 0, dist_2d, dist_3d

    elif iter > 200:
        print("Too slow")
        return True, -20, 0, dist_2d, dist_3d

    else:
        rwrd =  4 if dist_2d < p_2d_dist else -3
        rwrd += 2 if dist_3d < p_3d_dist else -1
        return False, rwrd, 0, dist_2d, dist_3d

#################################
##     -- 4  DIMENSIONS --     ##
#################################

def __action_4d(action):
    return [int(action[0]), int(action[1]), int(action[2]), 0, int(action[3])]   

def __reward_4d(observation, iter, p_2d_dist, p_3d_dist):
    dist_2d = LA.norm(observation[:2])
    dist_3d = LA.norm(observation[:3])

    print(observation[5])
    print(observation[6])
    
    bad_g = -5 if observation[5] > 0.001 or observation[6] > 0.001 else 2

    if observation[3] > 0.1 or observation[4] > 0.1:
        print("Collision")
        return True, dist_3d*1000 + bad_g, 0, dist_2d, dist_3d

    elif dist_3d > 0.1:
        print("Far in 3D")
        return True, -10 + bad_g, 0, dist_2d, dist_3d

    elif dist_2d > 0.08:
        print("Far in 2D")
        return True, -5 + bad_g, 0, dist_2d, dist_3d
    
    elif dist_2d < 0.005:
        if dist_3d <= 0.005:
            if observation[5] > 0.4 and observation[6] > 0.4:
                print("DONE :)")
                return True, 500 - iter, 1, dist_2d, dist_3d
                
            elif observation[5] > 0.1 or observation[6] > 0.1:
                print("Grasping")
                return False, 300 - iter, 0, dist_2d, dist_3d

            else:
                print("Not grasping")
                return False, 200 - iter, 0, dist_2d, dist_3d
        else:
            print("Above the cube")
            return False, 20 + bad_g, 0, dist_2d, dist_3d

    elif iter > 200:
        print("Too slow")
        return True, -20 + bad_g, 0, dist_2d, dist_3d

    else:
        rwrd =  10 if dist_2d < p_2d_dist else -3
        rwrd += 3  if dist_3d < p_3d_dist else -1
        rwrd += bad_g
        return False, rwrd, 0, dist_2d, dist_3d