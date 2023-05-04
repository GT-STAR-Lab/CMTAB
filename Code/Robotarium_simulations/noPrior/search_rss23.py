import matlab.engine
import numpy as np
import math

def search_task (X, Q, visualise):
    
    
    def intersectionArea(X1, Y1, R1, X2, Y2, R2) :

        Pi = 3.14

        # Calculate the euclidean distance
        # between the two points
        d = math.sqrt(((X2 - X1) * (X2 - X1)) + ((Y2 - Y1) * (Y2 - Y1)))
        if (d > R1 + R2) :
            ans = 0
        elif (d <= (R1 - R2) and R1 >= R2) :
            ans = floor(Pi * R2 * R2)
        elif (d <= (R2 - R1) and R2 >= R1) :
            ans = floor(Pi * R1 * R1)
        else :
            alpha = math.acos(((R1 * R1) + (d * d) - (R2 * R2)) / (2 * R1 * d)) * 2
            beta = math.acos(((R2 * R2) + (d * d) - (R1 * R1)) / (2 * R2 * d)) * 2

            a1 = (0.5 * beta * R2 * R2 ) - (0.5 * R2 * R2 * math.sin(beta))
            a2 = (0.5 * alpha * R1 * R1) - (0.5 * R1 * R1 * math.sin(alpha))
            ans = math.floor(a1 + a2)
        return ans

    def find_area(poses, search_robots, sensing_rad, whichspecies):
        #poses is a (3Xnumber of search robots array)
        #search_robots is (1Xnum_species array): how many robots of each species assigned for searching
        #whichspecies is (1Xnumber of search robots): species of each robot
        N = poses.shape[1]
        unionArea = 0
        coveredArea = 0
        for i in range (N):
            rad1 = sensing_rad[int(whichspecies[i])-1]
            coveredArea += math.pi * rad1 * rad1
            for j in range(i+1, N):
                rad2 = sensing_rad[int(whichspecies[j])-1]
                unionArea += intersectionArea(poses[0][i], poses[1][i], rad1,
                                               poses[0][j], poses[1][j], rad2)
        return coveredArea - unionArea
    X = np.ndarray.tolist(X)
    Q = np.ndarray.tolist(Q)
    
    eng = matlab.engine.start_matlab()
    search_robots = X[2]
    sensing_rad = [float(Q[0][3]), float(Q[1][3]), float(Q[2][3]), float(Q[3][3])] 
    speed = [float(Q[0][0]), float(Q[1][0]), float(Q[2][0]), float(Q[3][0])]
    #sensing_rad = [element * 0.07 for element in sensing_rad] #scaling based on sesning radius being 3, 2, 9, 4
    N = sum(search_robots)
    if N>1:
        output = eng.standardLloyd(float(N), search_robots, sensing_rad, visualise)
        output = np.array(output)
        poses = np.array(output[0:3][:])
        whichspecies = np.array(output[3][:])
        coveredArea = find_area(poses, search_robots, sensing_rad, whichspecies)
    else:
        for i in range(4):
            if sensing_rad[i]!=0:
                rad1 = sensing_rad[i]
        coveredArea = math.pi * rad1 * rad1
    totalArea = 3.2*2
    reward = coveredArea/totalArea
    if reward>1:
        reward = 1
    
    return reward

if __name__ ==  '__main__':
    
    X = [[1, 3, 3, 2],
         [1, 0, 0, 1],
         [1, 1, 4, 1]]
    '''
    Q = [[ 0.1, 4, 1, 0.21],
         [ 0.1, 1, 5, 0.14],
         [0.1, 1, 2, 0.63],
         [0.2, 2, 2, 0.28]]
    '''
    Q= np.array([[ 0.1, 4, 1, 0.18],
        [ 0.1, 1, 5, 0.12],
        [0.1, 1, 2, 0.54],
        [0.2, 2, 2, 0.24]])
    print (search_task(X, Q, True))
