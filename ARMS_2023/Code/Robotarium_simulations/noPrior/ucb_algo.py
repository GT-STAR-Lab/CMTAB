
def warn(*args, **kwargs):
    pass
import warnings
import time
warnings.warn = warn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import cvxpy
import scipy
import random
from exp_setup_GP import demos
from itertools import combinations_with_replacement
from itertools import permutations
from debris_rss23 import debris_task
from fire_rss23 import fire_task
from search_rss23 import search_task


####################################################################################################################
########################################   FUNCTION DEFINITIONS ######################################################
####################################################################################################################

class ucb_algo:
    
    def __init__(self, team, Q, num_species, num_tasks, num_traits, total_ite, delta):
        
        self.start_time = time.time()
        self.num_species = num_species
        self.num_tasks = num_tasks
        self.num_traits = num_traits
        #self.demo = demo
        self.og_team = team
        self.team = team
        self.Q = Q
        self.og_Q = Q
        self.csv_filename = "demos_robo.csv"
        #self.mu = self.demo.get_mu()
        #self.var = self.demo.get_var()
        self.T = 1000
        self.ite_to_run = total_ite
        self.delta = delta
        self.beta = np.zeros([1, self.num_tasks])
        
        self.low_margin = np.zeros([self.num_tasks, self.num_traits])
        self.high_margin = np.ones([self.num_tasks, self.num_traits])
        
    def pull_maxucb(self):
        
        self.scores = []
        self.highestArm = []
        self.best_armagg = np.zeros([self.num_tasks, self.num_traits])
        
        self.kernel0 = 1.0 * RBF(length_scale=1e-2, length_scale_bounds=(1e-2, 1e4)) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-4, 1e-1)) 
        self.kernel1 = 1.0 * RBF(length_scale=1e-2, length_scale_bounds=(1e-2, 1e4)) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-4, 1e-1))
        
        disc_int = 0.04 
        self.find_feasible_trait_space(5)
        #input = np.arange(np.min(self.low_margin), np.max(self.high_margin), disc_int)
        
        input = np.arange(0, 1, disc_int)
        
        meshgrid = np.array(np.meshgrid(input, input, input, input))
        self.X_grid = meshgrid.reshape(meshgrid.shape[0], -1).T
        #print("X_grid shape: ",self.X_grid.shape)
        
        #self.df = self.read_csvfile()
        
        self.assignments, self.taskScore, self.taskAssign = self.assigning_values()
        self.learned_prior_mesh = self.learn_prior_meshgrid()

        print("\nUCB Prior kernel: \n", self.learned_prior_mesh['kernel'][0], "\n",self.learned_prior_mesh['kernel'][1], "\n",self.learned_prior_mesh['kernel'][2])
        
        self.prior_lml = np.array(self.learned_prior_mesh['lml'])
        print("UCB Prior LML: ",self.learned_prior_mesh['lml'] )
        
        #print("\n\nScores calculated from scoring function as arms are pulled: ")

        self.pull_arm_n_learn()
        
        if self.highestArm:
            
            print("\nUCB: Allocations found for ", len(self.scores), " out of ", self.actual_ite, "iterations.")
            print("\nUCB Posterior kernel: \n", self.learned_prior_mesh['kernel'][0], "\n",self.learned_prior_mesh['kernel'][1], "\n",self.learned_prior_mesh['kernel'][2])
        
            self.posterior_lml = np.array(self.learned_prior_mesh['lml'])
            print("UCB Posterior LML: ",self.posterior_lml)

            self.inc_lml = 100*((self.posterior_lml - self.prior_lml)/self.prior_lml)

            #clearing memory
            del self.X_grid
            del self.assignments
            del self.taskScore
            del self.taskAssign
            del self.learned_prior_mesh
            del meshgrid
            print("\nHighest score in pulled arms(in simple UCB): ", np.max(np.array(self.highestArm)))
            print("Time taken by ucb algo:", round((time.time()- self.start_time)/60, 4))
            return self.scores, self.highestArm, self.inc_lml
        
        else: 
            print("\nPlain UCB could not find an allocation in given iterations.")
            return self.scores, self.highestArm, np.array([0, 0, 0])

    def int_least_squares_total(self, y):
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
                tol_fac= tol_fac-0.01
                x = np.zeros((self.num_tasks, self.num_species))
                if tol_fac<=0:
                    return np.round(x).astype('int')
            else:
                break

        return np.round(x.value).astype('int')


    def pull_arm_n_learn(self):

        arm  = np.zeros((self.taskAssign.shape[0], self.num_traits))
        ite = 0
        self.actual_ite = 0
        while ite <= self.ite_to_run and self.actual_ite < self.T-1:
            self.actual_ite+=1
            self.optimal_beta_selection(self.actual_ite)

            if (self.beta<0).any() or (self.beta==np.inf).any():
                continue
            else:
                temparg, maxucb = self.argmax_ucb(ite)
                #print(self.beta)
                
                for task in range(temparg.shape[0]):
                    arm[task] = self.X_grid[int(temparg[task])]
                    
                #print(arm)
                allocation = self.int_least_squares_total(arm)
                if (allocation>0).any():
                    ite+=1
                    trait_agg = (allocation@self.Q)/(self.team@self.Q)
                    score = self.get_task_score(allocation)
                    #print(score)
                    
                    if len(self.highestArm):
                        if np.sum(score) > self.highestArm[len(self.highestArm)-1]:
                            self.highestArm.append(np.sum(score))
                            self.best_armagg = trait_agg
                        else:
                            self.highestArm.append(self.highestArm[len(self.highestArm)-1])
                    else:
                        self.highestArm.append(np.sum(score))
                        self.best_armagg = trait_agg

                    self.scores.append(score)
                
                    self.taskAssign = np.insert(self.taskAssign, self.taskAssign.shape[1], trait_agg, axis= 1)
                    self.taskScore = np.insert(self.taskScore, self.taskScore.shape[1], score, axis= 1)
                    self.maxucb_learned_prior_mesh = self.learn_prior_meshgrid()
        print("Best trait aggregation in ucb algo:\n",self.best_armagg)
        
    
    def discretize_single_trait(self, num_int, lower_limit=0, upper_limit=1):
        '''
        lower_limit and upper_limit have dimensions: 1 X num_traits
        This function discretizes every trait for a single task
        '''
        input = []
        for trait in range(self.num_traits):
            input.append(np.arange(lower_limit[trait], upper_limit[trait], (upper_limit[trait]-lower_limit[trait])/num_int))
        meshgrid = np.array(np.meshgrid(input[0], input[1], input[2], input[3]))
        return (meshgrid.reshape(meshgrid.shape[0], -1).T)
    
    def find_feasible_trait_space(self, num_int):
        '''
        feasible_trait_space = np.zeros([(num_int**self.num_traits)**self.num_tasks, self.num_tasks, self.num_traits])
        trait_grids = np.zeros([ self.num_tasks, (num_int** self.num_traits),  self.num_traits])
        for task in range( self.num_tasks):
            trait_grids[task, :, :] =  self.discretize_single_trait( num_int, self.low_margin[task],  self.high_margin[task])

        input = np.arange(0, (num_int** self.num_traits), 1)
        meshgrid = np.array(np.meshgrid(input, input, input))
        self.X_grid = meshgrid.reshape(meshgrid.shape[0], -1).T
        for i in range( feasible_trait_space.shape[0]):
            for task in range( self.num_tasks):
                 feasible_trait_space[i, task, :] = trait_grids[task, self.X_grid[i, task] ,:] 
        indices = []
        #print("shape of feasible trait space before dropping confidence balls",  feasible_trait_space.shape)
        for i in range( feasible_trait_space.shape[0]):
            if (np.sum( feasible_trait_space[i], axis=0)<=1).all() and (feasible_trait_space[i]!=0).all():
                indices.append(i)
        feasible_trait_space =  feasible_trait_space[indices, :, :]
        self.low_margin = np.min(feasible_trait_space, axis=0)
        self.high_margin = np.max(feasible_trait_space, axis=0)
        '''
        feasible_trait_space = []
        trait_grids = np.zeros([ self.num_tasks, (num_int** self.num_traits),  self.num_traits])
        for task in range( self.num_tasks):
            trait_grids[task, :, :] =  self.discretize_single_trait( num_int, self.low_margin[task],  self.high_margin[task])

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
        self.low_margin = np.min(feasible_trait_space, axis=0)
        self.high_margin = np.max(feasible_trait_space, axis=0)
        
        
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
        '''
        for task in range(num_tasks):
            if task%2 != 0:
                beta.append( 2 * np.log(input_space_size * (ite ** 2) * (np.pi ** 2) / (3 * 10 * delta[task]))/5 )
            else:
                beta.append( 2 * np.log(input_space_size * ((T-ite) ** 2) * (np.pi ** 2) / (3 * 10 * delta[task]))/5 )
        '''
        
        for task in range(self.num_tasks):
            beta.append ((0.08 *np.log(((self.T-ite)**2)*2*(np.pi**2)/(3*self.delta[task]))) + (2*d*np.log(((self.T-ite)**2)*d*b*r*np.sqrt(np.log(4*d*a/self.delta[task])))))
        scaling = 5+((ite)*(10-5))/(self.T-ite)
        beta = np.array(beta)/ (2*(scaling**2))
        if ite%50 == 0:
            beta = beta/250
        #print("beta: ",np.sqrt(beta))
        self.beta = np.reshape(beta, (1, beta.shape[0]))

    def argmax_ucb(self, ite):
        """
        mu: list of length 3 (n X 1)
        sigma: list of length 3 (n X 1)
        beta: 1 x 3
        #Single Task Acquisition: return np.argmax(np.sum(temp, axis=0)), np.max(np.sum(temp, axis=0))
        """
        mu= self.learned_prior_mesh['mu']
        sigma= self.learned_prior_mesh['sigma']
        
        mu = np.reshape(np.array(mu), (3, mu[0].shape[0]))
        high_mu  = np.max(mu, axis=1).reshape(3,1)
        if (high_mu!=0).all():
            mu = mu/high_mu
        
        sigma = np.reshape(np.array(sigma), (3, sigma[0].shape[0]))
        high_sigma  = np.max(sigma, axis=1).reshape(3,1)
        if (high_sigma!=0).all():
            sigma = sigma/high_sigma
        
        beta = np.reshape(np.array(self.beta), (3, 1))
        t_ = np.zeros((sigma.shape[0], sigma.shape[1]))
        #print("sigma max",np.max(sigma, axis=1))
        #print("sigma min",np.min(sigma, axis=1))
        #print("mu max",np.max(mu, axis=1))
        #print("mu min:", np.min(mu, axis=1))
        for task in range(sigma.shape[0]):
            #print(sigma[task, 0:10])
            #print(mu[task, 0:10])
            #print(np.sqrt(beta))
            #t_[task,:] = sigma[task, :]*5*np.sqrt(beta)[task]
            t_[task,:] = sigma[task, :]*(random. randint(1, 5))*np.sqrt(beta)[task]
            #t_[task,:] = sigma[task, :]*beta[task]
        temp = (mu + t_)
        #temp.shape is 3 X points learned at
        
        argmax_ucb = np.argmax(temp, axis=1)
        '''
        if ite%3 == 0 and ite<=30:
            for task in range(temp.shape[0]):
                argmax_ucb[task] = random.randint(0, 390624)
        '''
        maxucb = np.max(temp, axis=1)
        #print("argmax_ucb: ", argmax_ucb)
        return argmax_ucb, maxucb
    
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
        df1.drop(df_[(df1["score1"]<0.01) | (df1["score2"]<0.01) | (df1["score3"]<0.01)].index, inplace = True)
        
        return df1
    

    def assigning_values(self):
        
        noOfPoints = 1

        assignments = np.zeros([noOfPoints, self.num_tasks, self.num_species])
        #qs = np.zeros([noOfPoints, self.num_species, self.num_traits])
        taskScore = np.zeros([self.num_tasks, noOfPoints])
        taskAssign = np.zeros([self.num_tasks, noOfPoints, self.num_traits])

        return assignments, taskScore, taskAssign


    def learn_prior_meshgrid(self):
        '''
        Learns prior for all tasks
        Returns a dictionary with keys: gp, kernel of learned prior, mu, sigma
        mu and sigma have mean and sigma for p0ints in the discretized traits' dimension space
        disc_int = Discretization interval
        '''

        prior = {'gp': [], 'kernel': [], 'mu':[], 'sigma':[], 'lml':[[],[],[]]}
        for task in range(self.num_tasks):
            if task == 2:
                prior['gp'].append(GaussianProcessRegressor(kernel=self.kernel1, normalize_y=True))
            else:
                prior['gp'].append(GaussianProcessRegressor(kernel=self.kernel0, normalize_y=True))
            prior['gp'][task].fit(self.taskAssign[task, :, :], self.taskScore[task, :])
            prior['kernel'].append(prior['gp'][task].kernel_)
            mu_, sigma_ = prior['gp'][task].predict(self.X_grid, return_std=True)
            #mu_, sigma_ = prior['gp'][task].predict(self.feasible_trait_space[:, task, :], return_std=True)
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

###################################################################################################################

def main():
    
    start_time = time.time()

    num_species = 4 #drone,rover,mini-rover,mini-drone
    num_tasks = 3  #move debris, search an environment, retrieve object from narrow passage
    num_traits = 3 #speed,payload,battery
    team = [7, 3, 4, 6]
    csv_filename = "demos_" + time.strftime("%m%d_%H%M%S")+ ".csv"
    num_demo = 1000 #approximate number of demos that will be generated
    add_noise=True
    randomize_team=True
    randomize_mu_var=True
    demo = demos(num_species, team, num_tasks, num_traits, csv_filename,
                 num_demo, add_noise=add_noise, random_team=True, random_mu_var=False)

    total_ite = 400
    delta = np.array([0.8, 0.8, 0.8]).T 
    
    experiment_baseline = groundtruth(num_species, num_tasks, num_traits, demo, total_ite, delta)
    ground_scores, ground_highscore = experiment_baseline.pull_maxucb()    

if __name__ == "__main__":
    main()

