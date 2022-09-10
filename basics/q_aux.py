def state2index(observ):
    '''Map an observation to an int
       IN:  obs -> distX, distY [-2, 2]
       OUT: index [101, 10100] '''

    x = (observ["distX"] + 2) * 25 * 100   # Map to [100, 10000]
    y = (observ["distY"] + 2) * 25         # Map to [1, 100]
    return int(x + y)

def action2index(action):
    ACTION_TABLE = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]]
    return ACTION_TABLE[action[0]+1][action[1]+1]

def index2action(index, dims):
    if dims >= 4:
        g = index % 3 - 1
        index = index // 3
    if dims >= 3:
        z = index % 3 - 1
        index = index // 3
    if dims >= 2:
        y = index % 3 - 1
        index = index // 3
        x = index % 3 - 1
    
    if dims == 2:
        return [x, y]
    if dims == 3:
        return [x, y, z]
    if dims == 4:
        return [x, y, z, g]
    else:
        return None

def actionFromAlg(state):
    '''Returns an action based on an algorithmic model'''
    action = []
    action.append(0 if abs(state["distX"]) < 0.001 else 1 if state["distX"] > 0.001 else -1)
    action.append(0 if abs(state["distY"]) < 0.001 else 1 if state["distY"] > 0.001 else -1)
    return action