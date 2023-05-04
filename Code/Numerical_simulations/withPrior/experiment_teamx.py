
def warn(*args, **kwargs):
    pass
import sys
import warnings
import time
warnings.warn = warn
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib import pylab as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import cvxpy
import scipy
import random
from exp_setup_GP import demos
from zoomin_all_tasks import stmria
from zoomin_ind_task import stmria_ind
from ucb_algo import ucb_algo
from itertools import combinations_with_replacement
from itertools import permutations


####################################################################################################################
########################################   FUNCTION DEFINITIONS ######################################################
####################################################################################################################


def calc_plot_mean_sigma(scores_all):
    
    #what is the largest number of arms pulled?
    largest_num_arms = 0
    for i in range(len(scores_all)):
        if scores_all[i].shape[0] > largest_num_arms:
            largest_num_arms = scores_all[i].shape[0] 
    
    #make size of all arrays equal so that they can be plotted
    for i in range(len(scores_all)):
        scores_t = np.zeros((largest_num_arms, 1))
        for j in range(scores_all[i].shape[0]):
            scores_t[j] = scores_all[i][j]
        t_ = scores_all[i][j]
        for k in range(scores_all[i].shape[0], largest_num_arms):
            scores_t[k] = t_
        scores_all[i] = scores_t
    scores_all = np.array(scores_all)
    
    #calculating mean and standard deviation of the scores
    scores_mean = np.mean(scores_all, axis=0)
    scores_std = np.std(scores_all, axis=0)
    scores_std2 = 2*(scores_std) # 2*sigma ~ 95% confidence region
    
    return scores_mean, scores_std2

def plotting_minscore(zoomin_all, ucb_all, rndm_all, figure_name):
    
    #ZOOMING ALGORITHM
    zoomin_mean, zoomin_std2 = calc_plot_mean_sigma(zoomin_all)
    
    plt.plot(np.arange(1,zoomin_mean.shape[0]+1), zoomin_mean, '-', color='black', label = "Zoom-in")

    plt.fill_between(np.arange(1,zoomin_mean.shape[0]+1), (zoomin_mean - zoomin_std2).reshape(zoomin_mean.shape[0],), (zoomin_mean + zoomin_std2).reshape(zoomin_mean.shape[0],),
                 color='blue', alpha=0.2)
    
    #RANDOM PULLING ALGORITHM
    rndm_mean, rndm_std2 = calc_plot_mean_sigma(rndm_all)
    
    plt.plot(np.arange(1,rndm_mean.shape[0]+1), rndm_mean, '-', color='red', label = "Random arms")

    plt.fill_between(np.arange(1,rndm_mean.shape[0]+1), (rndm_mean - rndm_std2).reshape(rndm_mean.shape[0],), (rndm_mean + rndm_std2).reshape(rndm_mean.shape[0],),
                 color='red', alpha=0.2)
    
    #BASELINE: UCB ALGORITHM
    
    ucb_mean, ucb_std2 = calc_plot_mean_sigma(ucb_all)

    plt.plot(np.arange(1,ucb_mean.shape[0]+1), ucb_mean, '-', color='green', label = "GP-UCB")

    plt.fill_between(np.arange(1,ucb_mean.shape[0]+1), (ucb_mean - ucb_std2).reshape(ucb_mean.shape[0],), (ucb_mean + ucb_std2).reshape(ucb_mean.shape[0],),
                     color='yellow', alpha=0.2)
    
    plt.legend()
    plt.title("Minimum task score as arms are pulled")
    plt.xlabel("Number of arms pulled")
    plt.ylabel("Min task score")
    plt.savefig(figure_name)
    plt.clf()

def high_so_far(arr):
    #Given an array of scores(some_numX1), returns an array with high performing score(seome_numX1)
    for i in range(1,arr.shape[0]):
        if arr[i]<=arr[i-1]:
            arr[i]=arr[i-1]
    return arr

def to_plot(list_scores, task, opt_score):
    list_ind = []
    list_indhigh = []
    for ite in range(len(list_scores)):
        list_ind.append((np.array(list_scores[ite][:,:,task]))/opt_score[task])
        list_t_ = np.array(list_scores[ite][:,:,task])
        list_ind_ = np.reshape(list_t_, (list_t_.shape[0],1))
        list_indhigh.append((high_so_far(list_ind_))/opt_score[task])
    return list_ind, list_indhigh    
    
def plotting_indscore(zoomin_ind_all, ucb_ind_all, rndm_ind_all, figure_name, optimal_score, num_tasks=3):

    optimal_score = optimal_score.reshape(3,1)       
    for task in range(num_tasks):
        zoomin_ind = []
        zoomin_indhigh = []
        ucb_ind = []
        ucb_indhigh = []
        rndm_ind = []
        rndm_indhigh = []
        zoomin_ind, zoomin_indhigh = to_plot(zoomin_ind_all, task, optimal_score)
        ucb_ind, ucb_indhigh = to_plot(ucb_ind_all, task, optimal_score)
        rndm_ind, rndm_indhigh = to_plot(rndm_ind_all, task, optimal_score)
        

        #ZOOMING ALGORITHM
        zoomin_mean, zoomin_std2 = calc_plot_mean_sigma(zoomin_ind)

        plt.plot(np.arange(1,zoomin_mean.shape[0]+1), zoomin_mean, '-', color='black', label = "Zoom-in")

        plt.fill_between(np.arange(1,zoomin_mean.shape[0]+1), (zoomin_mean - zoomin_std2).reshape(zoomin_mean.shape[0],), (zoomin_mean + zoomin_std2).reshape(zoomin_mean.shape[0],),
                     color='blue', alpha=0.2)

        #RANDOM PULLING ALGORITHM
        rndm_mean, rndm_std2 = calc_plot_mean_sigma(rndm_ind)

        plt.plot(np.arange(1,rndm_mean.shape[0]+1), rndm_mean, '-', color='red', label = "Random arms")

        plt.fill_between(np.arange(1,rndm_mean.shape[0]+1), (rndm_mean - rndm_std2).reshape(rndm_mean.shape[0],), (rndm_mean + rndm_std2).reshape(rndm_mean.shape[0],),
                     color='red', alpha=0.2)

        #BASELINE: UCB ALGORITHM
        
        ucb_mean, ucb_std2 = calc_plot_mean_sigma(ucb_ind)

        plt.plot(np.arange(1,ucb_mean.shape[0]+1), ucb_mean, '-', color='green', label = "GP-UCB")

        plt.fill_between(np.arange(1,ucb_mean.shape[0]+1), (ucb_mean - ucb_std2).reshape(ucb_mean.shape[0],), (ucb_mean + ucb_std2).reshape(ucb_mean.shape[0],),
                         color='yellow', alpha=0.2)

        plt.legend()
        plt.title("Task score as arms are pulled")
        plt.xlabel("Number of arms pulled")
        plt.ylabel("Task score")
        plt.savefig(figure_name+"task_"+str(task)+".png")
        plt.clf()
        
        #ZOOMING ALGORITHM: high individual
        zoomin_mean, zoomin_std2 = calc_plot_mean_sigma(zoomin_indhigh)

        plt.plot(np.arange(1,zoomin_mean.shape[0]+1), zoomin_mean, '-', color='blue', label = "Zoom-in")

        plt.fill_between(np.arange(1,zoomin_mean.shape[0]+1), (zoomin_mean - zoomin_std2).reshape(zoomin_mean.shape[0],), (zoomin_mean + zoomin_std2).reshape(zoomin_mean.shape[0],),
                     color='blue', alpha=0.2)

        #RANDOM PULLING ALGORITHM: high individual
        rndm_mean, rndm_std2 = calc_plot_mean_sigma(rndm_indhigh)

        plt.plot(np.arange(1,rndm_mean.shape[0]+1), rndm_mean, '-', color='red', label = "Random arms")

        plt.fill_between(np.arange(1,rndm_mean.shape[0]+1), (rndm_mean - rndm_std2).reshape(rndm_mean.shape[0],), (rndm_mean + rndm_std2).reshape(rndm_mean.shape[0],),
                     color='red', alpha=0.2)

        #BASELINE: UCB ALGORITHM: high individual
        ucb_mean, ucb_std2 = calc_plot_mean_sigma(ucb_indhigh)

        plt.plot(np.arange(1,ucb_mean.shape[0]+1), ucb_mean, '-', color='green', label = "GP-UCB")

        plt.fill_between(np.arange(1,ucb_mean.shape[0]+1), (ucb_mean - ucb_std2).reshape(ucb_mean.shape[0],), (ucb_mean + ucb_std2).reshape(ucb_mean.shape[0],),
                         color='yellow', alpha=0.2)

        plt.legend()
        plt.title("Task score as arms are pulled")
        plt.xlabel("Number of arms pulled")
        plt.ylabel("Task score")
        plt.savefig(figure_name+"taskhigh_"+str(task)+".png")
        plt.clf()
    

def plotting_realtimescore(zoomin_scores, ucb_scores, rndm_scores, optimal_score, high_demo, figure_name):
    
    #ZOOMING ALGORITHM
    zoomin_mean, zoomin_std2 = calc_plot_mean_sigma(zoomin_scores)
    
    plt.plot(np.arange(1,zoomin_mean.shape[0]+1), zoomin_mean, '--', color='blue', label = "Zoom-in")

    plt.fill_between(np.arange(1,zoomin_mean.shape[0]+1), (zoomin_mean - zoomin_std2).reshape(zoomin_mean.shape[0],), (zoomin_mean + zoomin_std2).reshape(zoomin_mean.shape[0],),
                 color='blue', alpha=0.2)

    
    #RANDOM PULLING ALGORITHM
    
    rndm_mean, rndm_std2 = calc_plot_mean_sigma(rndm_scores)
    
    plt.plot(np.arange(1,rndm_mean.shape[0]+1), rndm_mean, '--', color='red', label = "Random arms")

    plt.fill_between(np.arange(1,rndm_mean.shape[0]+1), (rndm_mean - rndm_std2).reshape(rndm_mean.shape[0],), (rndm_mean + rndm_std2).reshape(rndm_mean.shape[0],),
                 color='pink', alpha=0.2)

    #UCB ALGORITHM
    if ucb_scores:
        ucb_mean, ucb_std2 = calc_plot_mean_sigma(ucb_scores)
    
        plt.plot(np.arange(1,ucb_mean.shape[0]+1), ucb_mean, '--', color='green', label = "ucb arms")

        plt.fill_between(np.arange(1,ucb_mean.shape[0]+1), (ucb_mean - ucb_std2).reshape(ucb_mean.shape[0],), (ucb_mean + ucb_std2).reshape(ucb_mean.shape[0],), color='green', alpha=0.2)
        
        
        
        #ucb_scores = np.array(ucb_scores)
        #ucb_total = np.sum(ucb_scores, axis=2)
        #plt.plot(np.arange(1,ucb_total.shape[0]+1), ucb_total, 'g', label = "UCB")
    t_ = (np.arange(1,ucb_mean.shape[0]+1)).shape[0]
    if np.sum(optimal_score)>0.1:
        plt.plot(np.arange(1,ucb_mean.shape[0]+1), np.ones(t_)*np.sum(optimal_score), '--', color='black', label = "optimized on ground truth")

        #plt.plot(0, np.sum(optimal_score), marker="x", markersize=3, markeredgecolor="black", markerfacecolor="red", label="optimized on ground truth")
    plt.plot(np.arange(1,ucb_mean.shape[0]+1), np.ones(t_)*high_demo, '--', color='orange', label = "highest in demos")
    #plt.plot(0, high_demo, marker="o", markersize=3, markeredgecolor="black", markerfacecolor="red", label="highest in demos")

    plt.legend()
    plt.title("Total task score at each arm")
    plt.xlabel("Number of arms pulled")
    plt.ylabel("Total Score")
    plt.savefig(figure_name)
    plt.clf()
    

    
def plotting_high(zoomin_all, ucb_all, rndm_all, figure_name, optimal_score, high_demo, high_f):
    
    #ZOOMING ALGORITHM
    zoomin_mean, zoomin_std2 = calc_plot_mean_sigma(zoomin_all)
    
    plt.plot(np.arange(1,zoomin_mean.shape[0]+1), zoomin_mean, '-', color='black', label = "Zoom-in")

    plt.fill_between(np.arange(1,zoomin_mean.shape[0]+1), (zoomin_mean - zoomin_std2).reshape(zoomin_mean.shape[0],), (zoomin_mean + zoomin_std2).reshape(zoomin_mean.shape[0],),
                 color='blue', alpha=0.2)
    #plt.plot(zoomin_mean.shape[0], high_f, marker="p", markersize=3, markeredgecolor="black", markerfacecolor="red", label="highest in zoomed-in")

    #RANDOM PULLING ALGORITHM
    rndm_mean, rndm_std2 = calc_plot_mean_sigma(rndm_all)
    
    plt.plot(np.arange(1,rndm_mean.shape[0]+1), rndm_mean, '-', color='red', label = "Random arms")

    plt.fill_between(np.arange(1,rndm_mean.shape[0]+1), (rndm_mean - rndm_std2).reshape(rndm_mean.shape[0],), (rndm_mean + rndm_std2).reshape(rndm_mean.shape[0],),
                 color='red', alpha=0.2)
    
    #BASELINE: UCB ALGORITHM
    if ucb_all:
        ucb_mean, ucb_std2 = calc_plot_mean_sigma(ucb_all)

        plt.plot(np.arange(1,ucb_mean.shape[0]+1), ucb_mean, '-', color='green', label = "GP-UCB")

        plt.fill_between(np.arange(1,ucb_mean.shape[0]+1), (ucb_mean - ucb_std2).reshape(ucb_mean.shape[0],), (ucb_mean + ucb_std2).reshape(ucb_mean.shape[0],),
                     color='yellow', alpha=0.2)
    
    t_ = (np.arange(1,ucb_mean.shape[0]+1)).shape[0]
    if np.sum(optimal_score)>0.1:
        plt.plot(np.arange(1,ucb_mean.shape[0]+1), np.ones(t_)*np.sum(optimal_score), '--', color='black', label = "optimized on ground truth")
        #plt.plot(0, np.sum(optimal_score), marker="x", markersize=3, markeredgecolor="black", markerfacecolor="red", label="optimized on ground truth")
    plt.plot(np.arange(1,ucb_mean.shape[0]+1), np.ones(t_)*high_demo, '--', color='orange', label = "highest in demos")
    #plt.plot(0, high_demo, marker="o", markersize=3, markeredgecolor="black", markerfacecolor="red", label="highest in demos")
    
    plt.legend()
    plt.title("Highest total task score as arms are pulled")
    plt.xlabel("Number of arms pulled")
    plt.ylabel("highest Total Score")
    plt.savefig(figure_name)
    plt.clf()

def randomize_t(team):
    f_ = 2
    num_species = len(team)
    team_ = np.zeros(num_species)
    for i in range(num_species):
        if team[i]-f_ > 0:
            team_[i] = random.randint(team[i]-f_, team[i]+f_)
        else:
            team_[i] = random.randint(1, team[i]+f_)

    return team_.astype(int)

def randomize_Q(crafted_Q):
    num_species = crafted_Q.shape[0]
    num_traits = crafted_Q.shape[1]
    mu = np.mean(crafted_Q, axis=0)
    std = np.std(crafted_Q, axis=0)
    custom_Q = crafted_Q
    for i in range(num_traits):
        custom_Q[ :, i] = abs(np.random.normal(mu[i], 2*std[i], num_species))
        
    return custom_Q

def save_in_csv(scores, filename):
        '''
        Saves the task wise scores in a csv file
        '''
        rows = np.empty((scores.shape[0], 3))
        for i in range(scores.shape[0]):
            rows[i] = scores[i].flatten()
        fields = ['task1', 'task2', 'task3']
        
        with open(filename, 'w') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile) 
                
            # writing the fields 
            csvwriter.writerow(fields) 
                
            # writing the data rows 
            csvwriter.writerows(rows)
def get_team_dets(t):
    if t == 0:
        team = [7, 3, 4, 6]
        Q = np.array([[ 54,         374.19149698,  30.4       ],
                      [ 14,         437.82606374,  46.26      ],
                      [ 32,         403.87654746,  28.38      ],
                      [ 14,         320.2895844,   39.56      ]])
        optimal_alloc = np.array([[2, 0, 1, 2],
                                  [2, 1, 1, 2],
                                  [3, 2, 0, 0]])
        optimal_score = np.array([25.85448298, 29.78568611, 23.99808559])
    if t == 1:
        team = [7, 3, 5, 4]
        Q = np.array([[28.08265635, 363.1913876, 51.17452502],
                     [11.2092962, 434.2397925, 32.45088411],
                     [16.62072736, 378.5524796, 31.41285046],
                     [15.71146668, 437.0867929, 31.50725584]])
        optimal_alloc = np.array([[2, 0, 2, 1],
                                  [2, 0, 3, 1],
                                  [3, 1, 0, 1]])
        optimal_score = np.array([12.16805339, 14.62149975, 7.760210531])
    if t == 2:
        team = [7, 5, 5, 5]
        Q = np.array([[21.91620028, 334.3167349, 26.32679573],
                      [0.322888474, 420.5752322, 28.6841762],
                      [19.77116177, 411.8564803, 33.54407353],
                      [29.61508629, 318.1833611, 38.47998923]])
        optimal_alloc = np.array([[1, 0, 3, 1],
                                      [4, 1, 0, 2],
                                      [2, 0, 2, 2]])
        optimal_score = np.array([16.44619464, 23.13550629, 18.03430086])
    if t == 3:
        team = [8, 5, 2, 8]
        Q = np.array([[ 35.64434817, 360.2783504, 40.67676225],
                      [42.35537083, 379.7987193, 44.61184959],
                      [13.47245221, 385.0184488, 35.11826277],
                      [26.76332293, 341.3894082, 40.0623826]])
        optimal_alloc = np.array([[1, 3, 0, 0],
                                  [4, 0, 2, 0],
                                  [2, 2, 0, 1]])
        optimal_score = np.array([21.42224655, 26.65819992, 20.59109988])
    if t == 4:
        team = [5, 5, 6, 8]
        Q = np.array([[ 25.31495548, 419.1320611, 37.25159705],
                      [20.35733087, 452.9782712, 30.63383592],
                      [14.76300531, 388.0228638, 37.02369892],
                      [71.1834654, 340.6735854, 45.74954068]])
        optimal_alloc = np.array([[3, 0, 0, 1],
                                  [2, 1, 2, 1],
                                  [0, 2, 1, 2]])
        optimal_score = np.array([22.76690779, 26.57232919, 23.44936961])
    if t == 5:
        Q = np.array([[ 28.01327994, 401.1998586, 30.4497973],
                      [32.24055129, 398.9287471, 34.25438944],
                      [8.998191131, 276.0692059, 44.84298149],
                      [36.67407283, 404.8579071, 30.71812755]])
        team = [10, 3, 3, 5]
        optimal_alloc = np.array([[1, 0, 1, 3],
                                  [2, 3, 1, 0],
                                  [3, 0, 1, 2]])
        optimal_score = np.array([25.34841313, 28.05812906, 21.97387961])
    if t == 6:
        Q = np.array([[ 10.39877209, 449.1855699, 42.62998124],
                      [57.02527321, 362.2864568, 31.91467329],
                      [44.99613867, 427.1479489, 44.5305158],
                      [38.42744992, 502.6917495, 40.58526866]])
        team = [4, 5, 5, 5]
        optimal_alloc = np.array([[0, 0, 2, 2],
                                  [1, 0, 2, 2],
                                  [2, 2, 1, 0]])
        optimal_score = np.array([25.87692708, 29.43052661, 24.9747678])
    
    
    return team, Q, optimal_alloc, optimal_score
    
    
    
    
def main():
    args = sys.argv[1:]
    teamNo = int(args[0])
    print("team_num: ",teamNo )
    new_team, new_Q, optimal_alloc, optimal_score = get_team_dets(teamNo)
    print("The team is : ", new_team)
    print("The species-trait matrix Q is:\n ", new_Q)
    num_species = 4 #drone,rover,mini-rover,mini-drone
    num_tasks = 3  #move debris, search an environment, retrieve object from narrow passage
    num_traits = 3 #speed,payload,battery
    og_team = [7, 3, 4, 6]
    csv_filename = "demos" + str(teamNo)+"_"+time.strftime("%m%d_%H%M%S")+ ".csv"
    num_demo = 70 #approximate number of demos that will be generated
    add_noise=True
    randomize_team=False
    randomize_mu_var=False
    demo = demos(num_species, og_team, num_tasks, num_traits, csv_filename, teamNo, num_demo,  add_noise=add_noise, random_team=randomize_team, random_mu_var=randomize_mu_var)
    print("csv filename is: ",csv_filename )
    total_ite = 400
    delta = np.array([0.8, 0.8, 0.8]).T 

    
    
    rad_limit = 0.035
    start_time = time.time()
    
    high_f = 0
    lml_zoomin = []
    indlml_zoomin = []
    lml_ucb = []
    lml_rndm = []
    
    ucb_high_all = []
    ucb_scores_all = []
    ucb_ind_all = []
    ucb_min_all = []
    
    zoomin_high_all = []
    zoomin_scores_all = []
    zoomin_ind_all = []
    zoomin_min_all = []
    
    indzoomin_high_all = []
    indzoomin_scores_all = []
    indzoomin_ind_all = []
    indzoomin_min_all = []
    
    rndm_high_all=[]
    rndm_scores_all = []
    rndm_ind_all = []
    rndm_min_all = []

    num_rounds = 10
    for rollout in range(num_rounds):
        
        print("######################## ITERATION ", rollout+1, "\n")
        
        print("\n~~~ INDIVIDUAL ZOOM-IN ~~~ \n")
        
        experiment = stmria_ind(new_team, new_Q, num_species, num_tasks, num_traits, demo, total_ite, delta, rad_limit, start_time)
        indzoomin_scores, indzoomin_high, indhigh_demo, indhigh_f_, indlml_zoomin_ = experiment.zoomin_algo()
        indlml_zoomin.append(np.array(indlml_zoomin_))
        indzoomin_high_all.append(np.array(indzoomin_high))
        indzoomin_scores = np.array(indzoomin_scores)
        save_in_csv(indzoomin_scores, csv_filename[:-4]+"_i"+str(rollout)+".csv")
        
        indzoomin_ind_all.append(indzoomin_scores) #expect shape to be num of ite X (num_armsx1x3)
        indzoomin_scores_all.append(np.sum(indzoomin_scores, axis=2)) #check if len of zoomin_scores_all is number of iterations
        indzoomin_min_all.append(np.min((indzoomin_scores/optimal_score), axis=2))
        
        del experiment
        
        print("\n~~~ ZOOM-IN ~~~ \n")
        
        experiment = stmria(new_team, new_Q, num_species, num_tasks, num_traits, demo, total_ite, delta, rad_limit, start_time)
        #if rollout==0:         
            #optimal_alloc, optimal_score = experiment.optimise_groundtruth()
        zoomin_scores, zoomin_high, high_demo, high_f_, lml_zoomin_ = experiment.zoomin_algo()
        high_f +=high_f_
        lml_zoomin.append(np.array(lml_zoomin_))
        zoomin_high_all.append(np.array(zoomin_high))
        zoomin_scores = np.array(zoomin_scores)
        save_in_csv(zoomin_scores, csv_filename[:-4]+"_z"+str(rollout)+".csv")
        
        zoomin_ind_all.append(zoomin_scores) #expect shape to be num of ite X (num_armsx1x3)
        zoomin_scores_all.append(np.sum(zoomin_scores, axis=2)) #check if len of zoomin_scores_all is number of iterations
        zoomin_min_all.append(np.min((zoomin_scores/optimal_score), axis=2))
        
        del experiment
        print("\n~~~ GP-UCB ~~~ \n")
        #experiment_ucb = ucb_algo(new_team, new_Q, num_species, num_tasks, num_traits, demo, 10, delta)
        experiment_ucb = ucb_algo(new_team, new_Q, num_species, num_tasks, num_traits, demo, total_ite, delta)
        ucb_scores, ucb_high, lml_ucb_ = experiment_ucb.pull_maxucb()
        lml_ucb.append(np.array(lml_ucb_))
        ucb_scores = np.array(ucb_scores) 
        
        save_in_csv(ucb_scores, csv_filename[:-4]+"_u"+str(rollout)+".csv")
        
        ucb_ind_all.append(ucb_scores)
        ucb_scores_all.append(np.sum(ucb_scores, axis=2))
        ucb_min_all.append(np.min((ucb_scores/optimal_score), axis=2))
        if ucb_high:
            ucb_high_all.append(np.array(ucb_high))       
        del experiment_ucb
        
        print("\n~~~ RANDOM ARMS ~~~ \n")
        
        #experiment_rndm = stmria(new_team, new_Q, num_species, num_tasks, num_traits, demo, 10, delta, rad_limit, start_time)
        experiment_rndm = stmria(new_team, new_Q, num_species, num_tasks, num_traits, demo, total_ite, delta, rad_limit, start_time)
        rndm_scores, rndm_high, lml_rndm_ = experiment_rndm.baseline_random()
        lml_rndm.append(np.array(lml_rndm_))
        rndm_high_all.append(np.array(rndm_high))
        rndm_scores = np.array(rndm_scores) 
        
        save_in_csv(rndm_scores, csv_filename[:-4]+"_r"+str(rollout)+".csv")
        
        rndm_ind_all.append(rndm_scores)
        rndm_scores_all.append(np.sum(rndm_scores, axis=2))
        rndm_min_all.append(np.min((rndm_scores/optimal_score), axis=2))
        del experiment_rndm
        
       
    
        
        
    print("\nAllocation from optimizing on ground truth:\n", optimal_alloc)
    print("Score with the above allocation: ", optimal_score, "\nTotal score: ", np.sum(optimal_score))
    
    lml_avg_zoomin = np.average((np.array(lml_zoomin)).reshape(num_rounds, num_tasks), axis=0)
    indlml_avg_zoomin = np.average((np.array(indlml_zoomin)).reshape(num_rounds, num_tasks), axis=0)
    lml_avg_ucb = np.average((np.array(lml_ucb)).reshape(num_rounds, num_tasks), axis=0)
    lml_avg_rndm = np.average((np.array(lml_rndm)).reshape(num_rounds, num_tasks), axis=0)
    
    
    print("\nAverage increase in likelihood for zoom-in algorithm(%): ", lml_avg_zoomin)
    print("\nAverage increase in likelihood for indvidual zoom-in algorithm(%): ", indlml_avg_zoomin)
    print("\nAverage increase in likelihood for ucb algorithm(%): ", lml_avg_ucb)
    print("\nAverage increase in likelihood for random-arms algorithm(%): ", lml_avg_rndm)
    
    high_f /= num_rounds
    
    if ucb_high_all:
        plotting_high(zoomin_high_all, ucb_high_all, rndm_high_all, csv_filename[:-4]+"_high.png", optimal_score, high_demo, high_f)

if __name__ == "__main__":
    main()

