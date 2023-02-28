import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import time


def debris_task(X, Q, visualize= False):

    N = int(np.sum(X[1]))
    x = np.random.uniform(low=-1, high=1, size=(N,))
    y = np.random.uniform(low=-0.6, high=-1, size=(N,))
    z = np.zeros(N)
    init_cond_string = (str(x))[1:-1] + ";" + (str(y))[1:-1] +";" + (str(z))[1:-1] 
    initial_conditions = np.array(np.mat(init_cond_string))
    r = robotarium.Robotarium(number_of_robots=N, show_figure=visualize, initial_conditions=initial_conditions, sim_in_real_time=False)

    safety_radius = 0.17

    if visualize:
        rock = plt.imread('rock.png')
        x_img = np.linspace(0.0, 1, rock.shape[1])
        y_img = np.linspace(0.0, 1, rock.shape[0])

        rock_handle = r.axes.imshow(rock, extent=(-1.2, -0.8, 0.5, 0.9))

        start = plt.imread('start.png')
        x_img = np.linspace(0.0, 1, start.shape[1])
        y_img = np.linspace(0.0, 1, start.shape[0])

        start_handle = r.axes.imshow(start, extent=(-1, 1, -1, -0.7))

        box = plt.imread('box.png')
        x_img = np.linspace(0.0, 1, box.shape[1])
        y_img = np.linspace(0.0, 1, box.shape[0])

        box_handle = r.axes.imshow(box, extent=(1.2, 1.6, 0.6, -0.4))

        CM = np.random.rand(N,3) # Random Colors
        safety_radius_marker_size = determine_marker_size(r,safety_radius) # Will scale the plotted markers to be the diameter of provided argument (in meters)
        font_height_meters = 0.1
        font_height_points = determine_font_size(r,font_height_meters) # Will scale the plotted font height to that of the provided argument (in meters)

    x = r.get_poses()
    r.step()

    si_barrier_cert = create_single_integrator_barrier_certificate()
    si_position_controller = create_si_position_controller()
    si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()
    debris_cleared = False

    trait_values = []
    for i in range(4):
        for k in range(X[1][i]):
            trait_values.append(Q.T[2][i])
    trait_values = np.array(trait_values).flatten()

    speed_values = []
    for i in range(4):
        for k in range(X[1][i]):
            speed_values.append(Q.T[0][i])
    speed_values = np.array(speed_values).flatten()
    
    debris_reached = np.zeros(N)

    debris_point_1 = np.random.uniform(low=-1.2, high=-0.7, size=(N,))
    debris_point_2 = np.random.uniform(low=0.5, high=0.9, size=(N,))
    debris_point = np.array([debris_point_1,debris_point_2])


    clearing_point_1 = np.random.uniform(low=1.2, high=1.6, size=(N,))
    clearing_point_2 = np.random.uniform(low=-0.2, high=0.6, size=(N,))
    clearing_point = np.array([clearing_point_1,clearing_point_2])

    total_debris = 45
    reward_value = 45

    if visualize:    
        g = r.axes.scatter(x[0,:], x[1,:], s=safety_radius_marker_size, marker='o', facecolors='none',edgecolors='none',linewidth=7)
        text_counter = r.axes.text(1.4,0.9,str(total_debris),fontsize=20, color='k',fontweight='bold',horizontalalignment='right',verticalalignment='top',zorder=10)

    x_goal = np.copy(debris_point)

    iter_count = 0
    while(not debris_cleared):

        x = r.get_poses()

        iter_count+=1
        if iter_count > 2000:
            break 

        if visualize:
            g.set_offsets(x[:2,:].T)

        x_si = uni_to_si_states(x)

        status_changed = np.where(np.linalg.norm(x_goal-x_si,axis=0) < 0.1)[0]
        for i in status_changed:
            if debris_reached[i] == 1:
                total_debris -= trait_values[i]
                if total_debris < 0:
                    total_debris = 0
                if visualize:
                    text_counter.remove()
                    text_counter = r.axes.text(1.4,0.9,str(total_debris),fontsize=20, color='k',fontweight='bold',horizontalalignment='right',verticalalignment='top',zorder=10)
                #print("Debris needed to clear", str(total_debris))

            debris_reached[i] = 1-debris_reached[i] 

            if(debris_reached[i] == 0):
                x_goal[:,i] = debris_point[:,i]
            else:
                x_goal[:,i] = clearing_point[:,i]
            
            if (total_debris < 1) and (debris_reached[i] == 0):
                debris_cleared = True
        

        dxi = si_position_controller(x_si,x_goal)
        dxi = si_barrier_cert(dxi, x_si)
        dxu = si_to_uni_dyn(dxi, x)
        idx = np.where(speed_values<dxu[0])[0]
        dxu[0][idx] = speed_values[idx]
        r.set_velocities(np.arange(N), dxu)
        r.step()

    if debris_cleared:
        finished_caption = "TASK COMPLETED"
    else:
        finished_caption = "TASK TERMINATED. Ran for " + str(iter_count) +"iterations"

    if visualize:
        finished_label = r.axes.text(0,0,finished_caption,fontsize=font_height_points, color='k',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=20)

    #print(finished_caption)

    time.sleep(5)

    #Call at end of script to print debug information and for your script to run on the Robotarium server properly
    r.call_at_scripts_end()

    return ((reward_value-total_debris)/reward_value)

if __name__ ==  '__main__':
    X = np.array([[0,1,1,1],
     [0,2,2,2],
     [1,1,1,1]])
    '''
    Q= np.array([[ 0.1, 4, 1, 0.21],
        [ 0.1, 1, 5, 0.14],
        [0.1, 1, 2, 0.63],
        [0.2, 2, 2, 0.28]])
    '''
    Q= np.array([[ 0.1, 4, 1, 0.18],
        [ 0.1, 1, 5, 0.12],
        [0.1, 1, 2, 0.54],
        [0.2, 2, 2, 0.24]])
    reward = debris_task(X, Q, visualize= False)
    print(reward)
