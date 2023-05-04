
def warn(*args, **kwargs):
    pass
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
import os
from exp_setup_GP import demos
from ucb_algo import ucb_algo
from itertools import combinations_with_replacement
from itertools import permutations
from debris_rss23 import debris_task
from fire_rss23 import fire_task
from search_rss23 import search_task

####################################################################################################################
########################################   FUNCTION DEFINITIONS ######################################################
####################################################################################################################

class stmria_ind:
    
    def __init__(self, team, Q, num_species, num_tasks, num_traits, total_ite, delta, rad_limit, start_time):
        self.start_time = time.time()
        self.num_species = num_species
        self.num_tasks = num_tasks
        self.num_traits = num_traits
        #self.demo = demo
        #self.most_demo = self.demo.get_highest_score()
        self.og_team = team
        self.team = team
        self.Q = Q
        self.og_Q = Q
        self.csv_filename = "demos_robo.csv"
        #self.mu = self.demo.get_mu()
        #self.var = self.demo.get_var()
        self.rad_lim = rad_limit
        self.T = total_ite
        self.delta = delta
        self.beta = np.zeros([1, self.num_tasks])
        
        self.low_margin = np.zeros([self.num_tasks, self.num_traits])
        self.high_margin = np.ones([self.num_tasks, self.num_traits])
        self.feasible_allocations = []
        self.high_score_f_alloc = 0
        self.avg_score_f_alloc = 0
        
        #self.kernel0 = 1.0 * RBF(length_scale=1e-2, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1e-1, noise_level_bounds=(1e-15, 1e-6)) 
        #self.kernel1 = 1.0 * RBF(length_scale=1e-2, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1e-1, noise_level_bounds=(1e-15, 1e-6))
        
        self.kernel0 = 1.0 * RBF(length_scale=1e-2, length_scale_bounds=(1e-2, 1e4)) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-4, 1e-1)) 
        self.kernel1 = 1.0 * RBF(length_scale=1e-2, length_scale_bounds=(1e-2, 1e4)) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-4, 1e-1))
        

    def zoomin_algo(self, optimal_score=None):
        
        self.scores = []
        self.highestArm = []
        self.best_armagg = np.zeros([self.num_tasks, self.num_traits])
        
        self.zoomin_dict = []
        for task in range (self.num_tasks):
            #a list of three zoomin dictionaries, one for each task
            self.zoomin_dict.append({'ball_centres': [], 'conf_radius': [], 'num_pulled':[]})
        
        #self.low_margin, self.high_margin, self.feasible_trait_space = self.find_feasible_trait_space(5, self.low_margin, self.high_margin)
        
        disc_int = 0.2 #discretization interval
        input = np.arange(np.min(self.low_margin), np.max(self.high_margin), disc_int)
        meshgrid = np.array(np.meshgrid(input, input, input, input))
        self.X_grid = meshgrid.reshape(meshgrid.shape[0], -1).T
        
        self.df = self.read_csvfile()
        
        self.assignments, self.taskScore, self.taskAssign = self.assigning_values()
        self.highest_demo = np.max(np.sum(self.taskScore, axis=0))

        self.init_zoomin_dict()
        #self.arms_in_conf_ball()
        #print("\n\nScores calculated from scoring function as arms are pulled: ")
        self.learned_prior_mesh = self.learn_prior_meshgrid()

        print("\nZoomin Prior kernel: \n", self.learned_prior_mesh['kernel'][0], "\n",self.learned_prior_mesh['kernel'][1], "\n",self.learned_prior_mesh['kernel'][2])
        
        self.prior_lml = np.array(self.learned_prior_mesh['lml'])
        print("Zoomin Prior LML: ",self.prior_lml )
        
        self.select_arms_and_zoomin()
        print("\nZoomin Posterior kernel \n", self.learned_prior_mesh['kernel'][0], "\n",self.learned_prior_mesh['kernel'][1], "\n",self.learned_prior_mesh['kernel'][2])
        
        self.posterior_lml = np.array(self.learned_prior_mesh['lml'])
        print("Zoomin Posterior LML: ",self.posterior_lml)
        
        self.inc_lml = 100*((self.posterior_lml - self.prior_lml)/self.prior_lml)
        

        
        #clearing memory 
        #del self.feasible_trait_space
        del self.X_grid
        del self.assignments
        
        #del self.learned_prior_mesh
        del self.zoomin_dict
        del meshgrid
        
        #self.nest_zoomin()
        print("\nPicked_centres\n",self.best_armagg)
        print("\nFinding feasible allocations in zoomed-in space....\n")
        
        del self.taskScore
        del self.taskAssign
        '''
        self.feasible_allocs()
        print("\nNumber of feasible allocations: ", len(self.feasible_allocations))
        print("Highest score in these feasible allocations: ", self.high_score_f_alloc)
        print("Average score in these feasible allocations: ", self.avg_score_f_alloc)
        '''
        print("Highest score in pulled arms (zoom-in): ", np.max(np.array(self.highestArm)))
        print("Time taken in minutes:", round((time.time() - self.start_time)/60, 4))
        #self.plotting(optimal_score)
        del self.learned_prior_mesh
        return self.scores, self.highestArm, self.highest_demo, 0, self.inc_lml
        #highest_demo: highest score seen in "suboptimal demos"
        #most_demo: highest score seen in all demos (includes optimal demos)
        

    def get_task_score(self, x):
        '''
        x (allocation) should be of dimension (num_tasks, num_species)
        Assuming a ground truth for each task to be able to calculate/ predict the score 
        The ground truth is represented by the mua dn sigmas assumed for each task

        returns an array of scores of dimension : (num_tasks X 1)
        '''

        print(x)
        visualize = False
        score = np.zeros([1, self.num_tasks])
        
        score[0][0] = fire_task(x, self.Q, visualize)
        score[0][1] = debris_task(x, self.Q, visualize)
        score[0][2] = search_task(x, self.Q, visualize)
        
        return score


    def read_csvfile(self):
        '''
        Reads the data for demonstrations from csv file 
        Deletes demonstrations that have score of at least one task close to zero
        Returns the dataframe comprising of the demonstrations' data
        '''
        df_ = pd.read_csv(self.csv_filename)
        
        df1 = df_[['X_sp1_task1', 'X_sp2_task1', 'X_sp3_task1', 'X_sp4_task1', 
                   'X_sp1_task2', 'X_sp2_task2', 'X_sp3_task2','X_sp4_task2', 
                   'X_sp1_task3', 'X_sp2_task3', 'X_sp3_task3', 'X_sp4_task3', 
                   'Q_trait1_sp1', 'Q_trait2_sp1', 'Q_trait3_sp1', 'Q_trait4_sp1',
                   'Q_trait1_sp2', 'Q_trait2_sp2', 'Q_trait3_sp2', 'Q_trait4_sp2',
                   'Q_trait1_sp3', 'Q_trait2_sp3', 'Q_trait3_sp3', 'Q_trait4_sp3', 
                   'Q_trait1_sp4', 'Q_trait2_sp4', 'Q_trait3_sp4', 'Q_trait4_sp4', 
                   'score1', 'score2', 'score3', 'total_score']]
        
        #dropping scores that are too low: 
        #df1.drop(df_[(df1["score1"]<0.01) | (df1["score2"]<0.01) | (df1["score3"]<0.01)].index, inplace = True)
        
        return df1


    def assigning_values(self):
        
        noOfPoints = self.df.shape[0]

        assignments = np.zeros([noOfPoints, self.num_tasks, self.num_species])
        qs = np.zeros([noOfPoints, self.num_species, self.num_traits])
        taskScore = np.zeros([self.num_tasks, noOfPoints])
        taskAssign = np.zeros([self.num_tasks, noOfPoints, self.num_traits])
        
        for task in range(self.num_tasks):
            for species in range(self.num_species):
                assignments[:, task, species] = self.df['X_sp'+str(species+1)+'_task'+str(task+1)].to_list()
            taskScore[task, :] = self.df['score'+str(task+1)].to_list()
        
        for species in range(self.num_species):
            for trait in range(self.num_traits):
                qs[:, species, trait] = self.df['Q_trait'+str(trait+1)+'_sp'+str(species+1)].to_list()
                
        for task in range(self.num_tasks):
            for demo in range(noOfPoints):
                taskAssign[task, demo, :] = (assignments[demo, task, :]@self.Q) 

        yhat = self.team@self.Q

        for task in range(self.num_tasks):
            for demo in range(noOfPoints):
                taskAssign[task, demo, :] = (assignments[demo, task, :]@qs[demo, :, :]) /yhat
                
        return assignments, taskScore, taskAssign


    def learn_prior_meshgrid(self):
        '''
        Learns prior for all tasks
        Returns a dictionary with keys: gp, kernel of learned prior, mu, sigma
        mu and sigma have mean and sigma for points in the discretized traits' dimension space
        disc_int = Discretization interval
        '''

        prior = {'gp': [], 'kernel': [], 'mu':[], 'sigma':[], 'lml':[[],[],[]]}
        for task in range(self.num_tasks):
            prior['gp'].append(GaussianProcessRegressor(kernel=self.kernel0, normalize_y=True))
            prior['gp'][task].fit(self.taskAssign[task, :, :], self.taskScore[task, :])
            prior['kernel'].append(prior['gp'][task].kernel_)
            mu_, sigma_ = prior['gp'][task].predict(self.zoomin_dict[task]['ball_centres'], return_std=True)
            prior['mu'].append(mu_)
            prior['mu'][task]=np.reshape(np.array(prior['mu'][task]), (len(mu_),1))
            prior['sigma'].append(sigma_)
            prior['sigma'][task]=np.reshape(np.array(prior['sigma'][task]), (len(mu_),1))
            
            prior['lml'][task].append(prior['gp'][task].log_marginal_likelihood())
            
            #if covariance is required:
            #mu_, cov_ = prior['gp'][task].predict(taskAssign[task, :, :], return_cov=True)
            #prior['cov'].append(cov_)
            #prior['cov'][task]=np.array(prior['cov'][task])

        return prior


    def optimal_beta_selection(self, ite):
        """
        :param ite: the current round ite (this is 't' in old code, changed to reduce ambiguity)
        :param input_space_size: |D| of input space D.
        :param delta: hyperparameter delta where 0 < delta < 1, exclusively.
        shape of delta : num_tasks X 1
        :return: optimal beta for exploration_exploitation trade-off at round ite.
        """
        beta = []
        d = 5
        b= 1
        a=1
        r=1
        for task in range(self.num_tasks):
            beta.append ((0.08 *np.log(((self.T-ite)**2)*2*(np.pi**2)/(3*self.delta[task]))) + (2*d*np.log(((self.T-ite)**2)*d*b*r*np.sqrt(np.log(4*d*a/self.delta[task])))))
        scaling = 5+((ite)*(10-5))/(self.T-ite)
        beta = np.array(beta)/ (2*(scaling**2))
        
        self.beta= np.reshape(beta, (1, beta.shape[0]))


    def int_least_squares_total(self, y, optimize=False):
        """
        To compute the team given the trait requirements and Q
        Dimension of y is 3X3 when called for single task
        """
        tol_fac = 1
        #noOfTraits = Q.shape[1]
        yhat = self.team@self.Q
        ydenorm = np.multiply(y, yhat)
        #Dimension of ydenorm is 3X3
        
        while tol_fac>0: 
            x = cvxpy.Variable((self.num_tasks, self.num_species), integer=True)

            constraints = [cvxpy.sum(x[:,0]) <= self.team[0], cvxpy.sum(x[:,1]) <= self.team[1], cvxpy.sum(x[:,2]) <= self.team[2], cvxpy.sum(x[:,3]) <= self.team[3], x >= 0, (tol_fac*ydenorm) <= x@self.Q]

            obj = cvxpy.Minimize(cvxpy.pnorm(ydenorm - x@self.Q, 2)) 
            prob = cvxpy.Problem(obj, constraints)

            #prob.solve(solver = 'ECOS_BB')
            prob.solve(solver = cvxpy.CPLEX, reoptimize=True)

            #Use this condition when you are using ECOS_BB:  if prob.value == np.inf:
            if prob.value is None or prob.value == np.inf:
                x = np.zeros((self.num_tasks, self.num_species))
                if optimize:
                    tol_fac=tol_fac-0.01
                    if tol_fac<=0:
                        return np.round(x).astype('int')
                else:
                    return np.round(x).astype('int')
            else:
                break

        return np.round(x.value).astype('int')


    def discretize_single_trait(self, num_int, lower_limit=0, upper_limit=1):
        '''
        lower_limit and upper_limit have dimensions: 1 X num_traits
        This function discretizes every trait for a single task
        '''
        input = []
        for trait in range(self.num_traits):
            #lower_ = round(lower_limit[trait], 4)
            #upper_ = round(upper_limit[trait], 4)
            input.append(np.arange(lower_limit[trait], upper_limit[trait], (upper_limit[trait]-lower_limit[trait])/num_int))
            #input.append(np.arange(lower_, upper_, round(((upper_limit[trait]-lower_limit[trait])/num_int),4)))
        meshgrid = np.array(np.meshgrid(input[0], input[1], input[2], input[3]))
        return (meshgrid.reshape(meshgrid.shape[0], -1).T)

    def find_feasible_trait_space(self, num_int, low_margin, high_margin):
        feasible_trait_space = []
        #feasible_trait_space = np.zeros([(num_int**self.num_traits)**self.num_tasks, self.num_tasks, self.num_traits])
        trait_grids = np.zeros([ self.num_tasks, (num_int** self.num_traits),  self.num_traits])
        #if num_int ==3:
            #trait_grids = np.zeros([ self.num_tasks, 36,  self.num_traits])
        for task in range( self.num_tasks):
            trait_grids[task, :, :] =  self.discretize_single_trait( num_int, low_margin[task],  high_margin[task])

        input = np.arange(0, (num_int** self.num_traits), 1)
        meshgrid = np.array(np.meshgrid(input, input, input))
        self.X_grid = meshgrid.reshape(meshgrid.shape[0], -1).T
        for i in range( (num_int**self.num_traits)**self.num_tasks):
            temp = np.zeros([self.num_tasks, self.num_traits])
            for task in range( self.num_tasks):
                temp[task] = trait_grids[task, self.X_grid[i, task] ,:] 
            if (np.sum( temp, axis=0)<=1).all() and (temp!=0).all():
                feasible_trait_space.append(temp)
                #print("selected")
                
        feasible_trait_space = np.reshape(np.array(feasible_trait_space), (len(feasible_trait_space), self.num_tasks, self.num_traits))
        print("Shape of feasible trait space",  feasible_trait_space.shape)
        #self.low_margin = np.min(feasible_trait_space, axis=0)
        #self.high_margin = np.max(feasible_trait_space, axis=0)
        #self.feasible_trait_space = feasible_trait_space
        low_margin = np.min(feasible_trait_space, axis=0)
        high_margin = np.max(feasible_trait_space, axis=0)
        #self.feasible_trait_space = feasible_trait_space
        return low_margin, high_margin, feasible_trait_space

    def confidence_radius(self, i, num_pulled, init_rad=0.25):
        
        if num_pulled ==0:
            radius = init_rad
        else:
            radius = np.sqrt((2*np.log10(self.T))/(100*(num_pulled+1)))
        return radius


    def init_zoomin_dict(self):
        '''
        Dictionary to store activated arms and associated values
        '''
            
        for i in range(self.X_grid.shape[0]):
            for task in range(self.num_tasks):
                self.zoomin_dict[task]['ball_centres'].append(self.X_grid[i])
                self.zoomin_dict[task]['conf_radius'].append(self.confidence_radius(0,0))
                self.zoomin_dict[task]['num_pulled'].append(0) 
 
    def calc_max(self, mu, sigma, task):
            beta = np.reshape(np.array(self.beta), (3, 1))
            mu = np.reshape(np.array(mu), (1, len(self.zoomin_dict[task]['ball_centres'])))
            if (np.max(mu)!=0).all():
                mu = mu/np.max(mu)
            sigma = np.reshape(np.array(sigma), (1, len(self.zoomin_dict[task]['ball_centres'])))
            if (np.max(sigma)!=0).all():
                sigma = sigma/np.max(sigma)
            temp = sigma*(random. randint(1, 3))*np.sqrt(beta)[task]
            for i in range(sigma.shape[1]):
                temp[0][i]+= 2*(self.zoomin_dict[task]['conf_radius'][i])
            temp+= mu
            return np.argmax(temp), np.max(temp)
        
    def argmax_ucb(self):
        """
        mu: list of length 3 (n X 1)
        sigma: list of length 3 (n X 1)
        beta: 1 x 3
        #Single Task Acquisition: return np.argmax(np.sum(temp, axis=0)), np.max(np.sum(temp, axis=0))
        """
        
            
        argmax = []
        maxucb = []
        mu0= self.learned_prior_mesh['mu'][0]
        sigma0= self.learned_prior_mesh['sigma'][0]
        
        mu1= self.learned_prior_mesh['mu'][1]
        sigma1= self.learned_prior_mesh['sigma'][1]
        
        mu2= self.learned_prior_mesh['mu'][2]
        sigma2= self.learned_prior_mesh['sigma'][2]
        
        #each mu will be len(zoomin_dict[task]['ball_centres'])X1
        
        arg_, max_ = self.calc_max(mu0, sigma0, 0)
        argmax.append(arg_)
        maxucb.append(max_)
        
        arg_, max_ = self.calc_max(mu1, sigma1, 1)
        argmax.append(arg_)
        maxucb.append(max_)
        
        arg_, max_ = self.calc_max(mu2, sigma2, 2)
        argmax.append(arg_)
        maxucb.append(max_)
        #temp.shape is 3 X points learned at
        argmax = np.reshape(np.array(argmax),(3,1))
        maxucb = np.reshape(np.array(maxucb),(3,1))
        
        #print("max_ucb: ", np.max(temp, axis=1))
        return argmax, maxucb
    
    def select_arm(self, ite):
        arm  = np.zeros((self.taskAssign.shape[0], self.num_traits))
        #self.zoomin_dict['selection_index'] = []
        #self.zoomin_dict['selection_indexarg'] = []
        self.optimal_beta_selection(ite)
        #print(beta)
        temparg, maxucb = self.argmax_ucb()
        for task in range(temparg.shape[0]):
            #print("temparg[task]: ", temparg[task])
            #print(self.zoomin_dict[task]['ball_centres'][temparg[task][0]])
            arm[task] = self.zoomin_dict[task]['ball_centres'][temparg[task][0]]

        return temparg, arm
    
    def add_arm(self, new_centres, new_rad, task):
        self.zoomin_dict[task]['ball_centres'].append(new_centres)
        self.zoomin_dict[task]['conf_radius'].append(new_rad)
        self.zoomin_dict[task]['num_pulled'].append(0)
        
    def activate_more_arms(self, neighbour_arm, ite, task):
        new_rad = self.confidence_radius(ite, self.zoomin_dict[task]['num_pulled'][neighbour_arm])
        radius_diff = self.zoomin_dict[task]['conf_radius'][neighbour_arm] - new_rad

        new_centres = self.zoomin_dict[task]['ball_centres'][neighbour_arm]+(new_rad+(radius_diff/2))
        if (new_centres>=self.low_margin).all() and (new_centres<=self.high_margin).all():
            self.add_arm(new_centres, new_rad, task)
        new_centres = self.zoomin_dict[task]['ball_centres'][neighbour_arm]-(new_rad+(radius_diff/2))
        if (new_centres>=self.low_margin).all() and (new_centres<=self.high_margin).all():
            self.add_arm(new_centres, new_rad, task)
    
    def find_min_task(self):
        if self.scores:
            scores_ = (np.array(self.scores)).reshape(len(self.scores),self.num_tasks)
            mu = self.learned_prior_mesh['mu']
            mu = np.reshape(np.array(mu), (3, mu[0].shape[0]))
            high_mu  = np.max(mu, axis=1).reshape(3,1)
            scores_ = scores_/(high_mu.T)
            #print("len(scores):", len(self.scores), "; np.sum(scores_, axis=1): ",np.sum(scores_, axis=0) )
            return np.argmin(np.sum(scores_, axis=0))
        else:
            return 0
        
    def select_arms_and_zoomin(self):
        
        flag = False
        for ite in range(self.T):
            
            #arm, arm_agg = self.select_arm(ite, self.find_min_task()) #max-min optimization
            arm, arm_agg = self.select_arm(ite) #NORMAL OPTIMIZATION
            allocation = self.int_least_squares_total(arm_agg, True)
            #allocation_ = int_least_squares_total(zoomin_dict['ball_centres'][arm], Q, team)
            if (allocation != 0).any() and (np.sum(allocation, axis=1)!=0).all():
                print(allocation)
                print(arm_agg)
                for task in range(self.num_tasks):
                
                    rad_ = self.zoomin_dict[task]['conf_radius'][(arm[task][0])]
                    self.zoomin_dict[task]['num_pulled'][(arm[task])[0]] += 1
                    self.activate_more_arms((arm[task][0]), ite, task)
                    self.zoomin_dict[task]['conf_radius'][(arm[task][0])] = self.confidence_radius(ite, self.zoomin_dict[task]['num_pulled'][(arm[task][0])])                    
                    
                trait_agg = (allocation@self.Q)/(self.team@self.Q)

                score = self.get_task_score(allocation)
                #print(score)

                if len(self.highestArm):
                    if np.sum(score) > self.highestArm[len(self.highestArm)-1]:
                        self.highestArm.append(np.sum(score))
                        self.best_armagg = trait_agg
                        self.best_rad = rad_
                        best_arm = arm
                    else:
                        self.highestArm.append(self.highestArm[len(self.highestArm)-1])
                else:
                    self.highestArm.append(np.sum(score))
                    self.best_armagg = trait_agg
                    self.best_rad = rad_
                    best_arm = arm

                self.scores.append(score)
                self.taskAssign = np.insert(self.taskAssign, self.taskAssign.shape[1], trait_agg, axis= 1)
                self.taskScore = np.insert(self.taskScore, self.taskScore.shape[1], score, axis= 1)
                self.learned_prior_mesh = self.learn_prior_meshgrid()
            else:
                for task in range(self.num_tasks):
                    self.zoomin_dict[task]['num_pulled'][(arm[task])[0]] += 1
                    self.activate_more_arms((arm[task][0]), ite, task)                
                    self.zoomin_dict[task]['conf_radius'][(arm[task][0])] = 0
                    self.learned_prior_mesh = self.learn_prior_meshgrid()
            for task in range(self.num_tasks):
                if any((x <= self.rad_lim and x > 0) for x in self.zoomin_dict[task]['conf_radius']) :
                #if self.best_rad <= self.rad_lim :
                    flag = True
                    print("\nAllocations found for ", len(self.scores), " out of ", ite+1, " iterations")
                    print("Loop broken with best radius: ", self.best_rad)
                    print("Loop broken when confidence radius lesser than ", self.rad_lim)
            if flag:
                break
        for ite_ in range(ite, self.T-1):
            
            #arm, arm_agg = self.select_arm(ite, self.find_min_task()) #max-min optimization
            arm, arm_agg = self.select_arm(ite_) #NORMAL OPTIMIZATION
            allocation = self.int_least_squares_total(arm_agg, True)
            #allocation_ = int_least_squares_total(zoomin_dict['ball_centres'][arm], Q, team)
            if (allocation != 0).any():
                
                trait_agg = (allocation@self.Q)/(self.team@self.Q)

                score = self.get_task_score(allocation)
                #print(score)

                if len(self.highestArm):
                    if np.sum(score) > self.highestArm[len(self.highestArm)-1]:
                        self.highestArm.append(np.sum(score))
                        self.best_armagg = trait_agg
                        self.best_rad = rad_
                        best_arm = arm
                    else:
                        self.highestArm.append(self.highestArm[len(self.highestArm)-1])
                else:
                    self.highestArm.append(np.sum(score))
                    self.best_armagg = trait_agg
                    self.best_rad = rad_
                    best_arm = arm

                self.scores.append(score)
                self.taskAssign = np.insert(self.taskAssign, self.taskAssign.shape[1], trait_agg, axis= 1)
                self.taskScore = np.insert(self.taskScore, self.taskScore.shape[1], score, axis= 1)
                self.learned_prior_mesh = self.learn_prior_meshgrid()

    def combinations_per_species(self, num_robots):
        comb = list(combinations_with_replacement(list(range(num_robots+1)), self.num_tasks))
        comb = [ elem for elem in comb if 0<np.sum(np.array(elem))<=num_robots ]
        all_comb = []
        for elem in comb:
            perm = permutations(elem)
            all_comb.append(list(set(list(perm))))
        return np.array([x for xs in all_comb for x in xs])

    def feasible_allocs(self):
        high_feasible_score = 0
        avg_score = 0
        rad_scale = 1
        while len(self.feasible_allocations)==0 and rad_scale<=2.25:
            
            low_margin = self.best_armagg - (rad_scale*self.rad_lim)
            high_margin = self.best_armagg + (rad_scale*self.rad_lim)
            yhat = self.team@self.Q
            Q_ = self.Q/yhat
            allocPerSpecies = {'species1':[], 'species2':[], 'species3':[], 'species4':[]}
            for num_robots in range(self.num_species):
                allocPerSpecies['species'+str(num_robots+1)]= self.combinations_per_species(self.team[num_robots])
                #print(allocPerSpecies['species'+str(num_robots+1)].shape)
                #for current team  the shapes are: 119, 19, 34, 83

            zoomed_traitspace_ = []
            for species1 in allocPerSpecies['species1']:
                for species2 in allocPerSpecies['species2']:
                    for species3 in allocPerSpecies['species3']:
                        for species4 in allocPerSpecies['species4']:
                            alloc_ = np.stack((species1, species2, species3, species4), axis=-1)

                            if (low_margin<=alloc_@Q_).all() and (high_margin>=alloc_@Q_).all():
                                self.feasible_allocations.append(alloc_)
                                zoomed_traitspace_.append(alloc_@Q_)
                                avg_score+= np.sum(self.get_task_score(alloc_))
                                if np.sum(self.get_task_score(alloc_))>high_feasible_score:
                                    high_feasible_score = np.sum(self.get_task_score(alloc_))

            rad_scale+=0.1
        
        self.high_score_f_alloc = high_feasible_score
        self.avg_score_f_alloc = avg_score/len(self.feasible_allocations)
        
        
        #print("\nRadius scaled by: ", rad_scale-0.25)
        zoomed_traitspace = (np.array(zoomed_traitspace_)).reshape(len(zoomed_traitspace_), self.num_tasks, self.num_traits)
        mu = np.zeros((zoomed_traitspace.shape[0], self.num_tasks))
        sigma = np.zeros((zoomed_traitspace.shape[0], self.num_tasks))
        for task in range(self.num_tasks):
            mu[:, task], sigma[:, task] = self.learned_prior_mesh['gp'][task].predict(zoomed_traitspace[:,task,:], return_std=True)
            
        mu = mu/np.max(mu, axis=0)
        sigma = sigma/np.max(sigma, axis=0)
        finding_best_alloc=True
        tries=0
        bestest_ever = np.zeros((self.num_tasks, self.num_species))
        while finding_best_alloc and tries<zoomed_traitspace.shape[0]/2:
            tries+=1
            index= np.argmax(np.sum(mu, axis=1)+np.sum(sigma, axis=1))
            alloc_ = self.feasible_allocations[index]
            score = self.get_task_score(alloc_)
            self.scores.append(score)
            if np.sum(score) > self.highestArm[len(self.highestArm)-1]:
                #print("Found better score in feasible allocations: ", np.sum(score))
                self.highestArm.append(np.sum(score))
                self.best_armagg = zoomed_traitspace[index]
                bestest_ever = alloc_
                for task in range(self.num_tasks):
                    mu[index][task]=0
                    sigma[index][task]=0
            else:
                self.highestArm.append(self.highestArm[len(self.highestArm)-1])
                #print("didn't find better in feasible allocations: ", np.sum(score))
                finding_best_alloc=False
        
        print("Best arm aggregation in zoom-in algorithm:\n", self.best_armagg)
              
    
    
    def optimise_groundtruth(self):

        low_margin = np.zeros([self.num_tasks, self.num_traits])
        high_margin = np.ones([self.num_tasks, self.num_traits])
        high_score = 0
        best_alloc = np.zeros([self.num_tasks, self.num_species])
        allocPerSpecies = {'species1':[], 'species2':[], 'species3':[], 'species4':[]}
        yhat = self.team@self.Q
        Q_ = self.Q/yhat
        allocPerSpecies = {'species1':[], 'species2':[], 'species3':[], 'species4':[]}
        for num_robots in range(self.num_species):
            allocPerSpecies['species'+str(num_robots+1)]= self.combinations_per_species(self.team[num_robots])
            #print(allocPerSpecies['species'+str(num_robots+1)].shape)
            #for current team  the shapes are: 119, 19, 34, 83

        zoomed_traitspace_ = []
        for species1 in allocPerSpecies['species1']:
            for species2 in allocPerSpecies['species2']:
                for species3 in allocPerSpecies['species3']:
                    for species4 in allocPerSpecies['species4']:
                        alloc_ = np.stack((species1, species2, species3, species4), axis=-1)

                        if (low_margin<=alloc_@Q_).all() and (high_margin>=alloc_@Q_).all():
                            if np.sum(self.demo.get_task_score_gp(alloc_, self.Q)) > high_score:
                                high_score = np.sum(self.demo.get_task_score_gp(alloc_, self.Q)) 
                                best_alloc = alloc_
        return best_alloc, self.demo.get_task_score_gp(best_alloc, self.Q)
    
    def learn_prior_meshgrid_rand(self):
        '''
        Learns prior for all tasks
        Returns a dictionary with keys: gp, kernel of learned prior, mu, sigma
        mu and sigma have mean and sigma for points in the discretized traits' dimension space
        disc_int = Discretization interval
        '''

        prior = {'gp': [], 'kernel': [], 'mu':[], 'sigma':[], 'lml':[[],[],[]]}
        for task in range(self.num_tasks):
            prior['gp'].append(GaussianProcessRegressor(kernel=self.kernel0, normalize_y=True))
            prior['gp'][task].fit(self.taskAssign[task, :, :], self.taskScore[task, :])
            prior['kernel'].append(prior['gp'][task].kernel_)
            mu_, sigma_ = prior['gp'][task].predict(self.X_grid, return_std=True)
            prior['mu'].append(mu_)
            prior['mu'][task]=np.reshape(np.array(prior['mu'][task]), (len(mu_),1))
            prior['sigma'].append(sigma_)
            prior['sigma'][task]=np.reshape(np.array(prior['sigma'][task]), (len(mu_),1))
            
            prior['lml'][task].append(prior['gp'][task].log_marginal_likelihood())
            
            #if covariance is required:
            #mu_, cov_ = prior['gp'][task].predict(taskAssign[task, :, :], return_cov=True)
            #prior['cov'].append(cov_)
            #prior['cov'][task]=np.array(prior['cov'][task])

        return prior
    
    
    def plotting(self, optimal_score=None):
        scores = np.array(self.scores)
        total_score=np.sum(scores, axis=2)
        highscore = np.array(self.highestArm)
        
        maxucb_scores = np.array(self.maxucb_scores)
        maxucb_total_score=np.sum(maxucb_scores, axis=2)
        maxucb_highscore = np.array(self.maxucb_highestArm)
        if optimal_score is not None:
            plt.plot(0, np.sum(optimal_score), marker="x", markersize=3, markeredgecolor="black", markerfacecolor="red")
        plt.plot(0, self.highest_demo, marker="o", markersize=3, markeredgecolor="black", markerfacecolor="red")
        plt.plot(np.arange(1,scores.shape[0]+1), total_score, 'r', label = "total score")
        plt.plot(np.arange(1,highscore.shape[0]+1), highscore, 'b', label = "High score till t")
        plt.legend()
        plt.title("Total task scores as arms are pulled \n(The dot is highest score recorded in expert demos) ")
        plt.xlabel("Number of arm pulled")
        plt.ylabel("Total Score")
        plt.show()
        

        
