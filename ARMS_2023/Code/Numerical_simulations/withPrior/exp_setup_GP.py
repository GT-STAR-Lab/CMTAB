
import numpy as np
import csv
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
import os
import sys
import random
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

traits = ["speed","payload","battery"]
crafted_Q = np.array([[ 54,         374.19149698,  30.4       ],
                      [ 14,         437.82606374,  46.26      ],
                      [ 32,         403.87654746,  28.38      ],
                      [ 14,         320.2895844,   39.56      ]])

class demos:

    def __init__(self, num_species, num_agents_per_species, num_tasks, num_traits,
                 csv_filename, teamNo, num_demo=None, random_Q=None, add_noise=None, random_team=None, random_mu_var=None):
        self.num_species = num_species
        self.og_num_agents_per_species = num_agents_per_species
        self.num_agents_per_species = num_agents_per_species
        self.num_tasks = num_tasks
        self.num_traits = num_traits
        self.csv_filename = csv_filename
        self.teamNo = teamNo
        if num_demo is None:
            self.num_demo = 6000
        else:
            #From total demonstrations, only about 1/6th are selected based on a sub-optimality condition
            self.num_demo = num_demo*6
        if random_Q is None:
            self.randomize_Q = False
        else:
            self.randomize_Q = random_Q
        if add_noise is None:
            self.add_noise = False
        else:
            self.add_noise = add_noise
        if random_team is None:
            self.num_agents_per_species = self.og_num_agents_per_species
            self.random_team = False
        else:
            self.random_team = random_team
            if self.random_team is True:
                self.randomize_team()
                
        self.mu = np.array([[  158.67580604,    1816.80229093,  168.40949543],
                            [  177.84615385,    2221.62541775,  212.90611196],
                            [   183.53351104 ,  2092.25017599,  193.70252983]])
        
        self.var = np.array([[ 2.49129513e+03,  2.57649083e+05, 2.27535208e+03],
                             [ 2.44002190e+03,  2.33179563e+05, 1.98966750e+03],
                             [ 2.55986858e+03, 2.64952096e+05, 2.36640265e+03]])
        if random_mu_var is None:
            self.random_mu_var = False
        else:
            self.random_mu_var = random_mu_var
            if self.random_mu_var is True:
                self.randomize_mu_var()
        
        self.X_test = []
        self.custom_Q = np.zeros((self.num_species,self.num_traits))
        self.Q_test = []
        self.sub_optimal_demos = {}
        self.task_wise_scores = []
        self.create_demos()
        self.save_in_csv()
        
    def set_Q(self, rndm_q_demo=False):
        crafted_Q = np.array([[ 54,         374.19149698,  30.4       ],
                              [ 14,         437.82606374,  46.26      ],
                              [ 32,         403.87654746,  28.38      ],
                              [ 14,         320.2895844,   39.56      ]])
        mu = np.mean(crafted_Q, axis=0)
        std = np.std(crafted_Q, axis=0)
        if rndm_q_demo is True:
            for i in range(self.num_traits):
                self.custom_Q[ :, i] = abs(np.random.normal(mu[i], 2*std[i], self.num_species))
        else:
            self.custom_Q = crafted_Q
            
    def get_mu(self):
        return self.mu
    
    def get_var(self):
        return self.var
    
    def get_noise(self):
        return np.random.normal(np.zeros((1,self.num_tasks)), self.sigma_noise, [1, self.num_tasks])
    
    def get_Q(self):
        return self.custom_Q 
    
    def get_filename(self):
        #return "demos_team"+str(self.teamNo)+".csv"
        return self.csv_filename
    
    def get_team(self):
        return self.num_agents_per_species

    def get_X(self):
        X_test = np.zeros((self.num_tasks, self.num_species))
        for s in range(self.num_species):
            R = np.random.choice(range(self.num_tasks+1), size=self.num_agents_per_species[s])
            for m in range(self.num_tasks):
                X_test[m, s] = np.sum(R == m)
            X_test = X_test.round()
        X_test = X_test.astype(np.int32)
        return X_test

    def func(self, y_task, sigma_inv, mu):
        '''
        y_task and mu's dimesnion: (1 X num_traits)
        '''
        y_task = y_task - mu
        return (np.dot(y_task , np.dot(sigma_inv, y_task.T))) 
        

    def get_task_score_gp(self, x, Q=None):
        '''
        x (allocation) should be of dimension (num_tasks, num_species)
        Assuming a ground truth for each task to be able to calculate/ predict the score 
        The ground truth is represented by the mua dn sigmas assumed for each task

        returns an array of scores of dimension : (num_tasks X 1)
        '''
        sigma_task1 = np.zeros((self.num_traits, self.num_traits))
        sigma_task2 = np.zeros((self.num_traits, self.num_traits))
        sigma_task3 = np.zeros((self.num_traits, self.num_traits))
        
        #Go the below values from certain score functions run earlier

        mu_task1 = np.array(self.mu[0]).reshape(1,self.num_traits)
        mu_task2 = np.array(self.mu[1]).reshape(1,self.num_traits) 
        mu_task3 = np.array(self.mu[2]).reshape(1,self.num_traits)

        var_task1 = np.array(self.var[0]).reshape(1,self.num_traits)
        var_task2 = np.array(self.var[1]).reshape(1,self.num_traits)
        var_task3 = np.array(self.var[2]).reshape(1,self.num_traits)
        
        np.fill_diagonal(sigma_task1, var_task1)
        np.fill_diagonal(sigma_task2, var_task2)
        np.fill_diagonal(sigma_task3, var_task3)

        sigma_inv1 = np.linalg.inv(sigma_task1)
        sigma_inv2 = np.linalg.inv(sigma_task2)
        sigma_inv3 = np.linalg.inv(sigma_task3)
        
        if Q is None:
            y_task = x@self.custom_Q #Dimension : noOfTasks X noOfTraits
        else:
            y_task = x@Q
        score1 = (1/(2*np.pi*np.sqrt(np.linalg.det(sigma_task1)))) * np.exp(-0.5*self.func(y_task[0], sigma_inv1, mu_task1))
        score2 = (1/(2*np.pi*np.sqrt(np.linalg.det(sigma_task2)))) * np.exp(-0.5*self.func(y_task[1], sigma_inv2, mu_task2))
        score3 = (1/(2*np.pi*np.sqrt(np.linalg.det(sigma_task3)))) * np.exp(-0.5*self.func(y_task[2], sigma_inv3, mu_task3))
        score = np.array([score1, score2, score3])
        score = score*(1e8)*2
        
        return np.reshape(score,(1,self.num_tasks))

    def get_indices_score_80percent(self):
        #array should be shape "some_number X 1"
        array = self.sub_optimal_demos['Total_score']
        select_indices = []
        array.sort()
        for i in range(int(0.8*array.shape[0])):
            select_indices.append(i)
        return np.array(select_indices)
    
    def get_indices_score_stddevcase1(self):
        #array should be shape "some_number X 1"
        array = self.sub_optimal_demos['Total_score']
        select_indices = []
        for i in range(array.shape[0]):
            if array[i,0]>=(np.mean(array)) and array[i,0]<=(np.mean(array)+(2*np.std(array))):
                select_indices.append(i)
        return np.array(select_indices)
    
    def get_indices_score_stddevcase2(self):
        #array should be shape "some_number X 1"
        array = self.sub_optimal_demos['Total_score']
        select_indices = []
        for i in range(array.shape[0]):
            if array[i,0]>=(np.mean(array)+np.std(array)) and array[i,0]<=(np.mean(array)+(2*np.std(array))):
                select_indices.append(i)
        return np.array(select_indices)

    def get_custom_Q(self):
        return self.custom_Q

    def get_csv_name(self):
        return self.csv_filename
    
    def get_highest_score(self):
        return self.highest_score

    def create_demos(self):
        ################ FOR FIXED TEAM AND Q IN EVERY DEMO ##################
        #Generate a species-trait matrix
        self.set_Q(self.randomize_Q)

        #Generate random allocations
        for i in range(self.num_demo):
            x_i = self.get_X()
            self.X_test.append(x_i)
            
            q_i = self.get_Q()
            self.Q_test.append(q_i)
        self.X_test = np.array(self.X_test)
        self.Q_test = np.array(self.Q_test)
        
        '''
        ################ FOR VARYING TEAM AND VARYING Q IN EVERY DEMO ##################

        #Generate random allocations
        for i in range(self.num_demo):
            self.randomize_team()
            x_i = self.get_X()
            self.X_test.append(x_i)
            
            self.set_Q(True)
            q_i = self.get_Q()
            self.Q_test.append(q_i)
        self.X_test = np.array(self.X_test)
        self.Q_test = np.array(self.Q_test)
        '''
        #Get the scores for each task for each allocation
        for demo in range(self.num_demo):
            task_wise = self.get_task_score_gp(self.X_test[demo])
            self.task_wise_scores.append(task_wise)
        self.task_wise_scores = np.array(self.task_wise_scores)
        #task_wise_scores.shape: num of demos X 1 X num of tasks
        self.sum_task_scores = np.sum(self.task_wise_scores, axis =2)

        #Store the values in a dictionary
        self.sub_optimal_demos = {'X': self.X_test, 'Q': self.Q_test, 'Task_scores':self.task_wise_scores, 'Total_score':self.sum_task_scores}
        #Select the demonstrations based on a condition for sub-optimality
        #indices = self.get_indices_score_80percent()
        #indices = self.get_indices_score_stddevcase1()
        self.highest_score = np.max(self.sum_task_scores)
        #print("Best score in demos(including optimal):\n",self.highest_score )
        #print("Allocation for this best score:\n",self.sub_optimal_demos['X'][np.argmax(self.sum_task_scores)] )
        t_ = self.sub_optimal_demos['X'][np.argmax(self.sum_task_scores)]@self.sub_optimal_demos['Q'][np.argmax(self.sum_task_scores)]
        for i in range(self.num_tasks):
            t_[i]=t_[i]/(self.num_agents_per_species@self.sub_optimal_demos['Q'][np.argmax(self.sum_task_scores)])
        #print("Trait aggregation with this allocation:\n", t_)
        indices = self.get_indices_score_stddevcase2()
        print("Number of sub-optimal demos: ", indices.shape[0])
        for key in self.sub_optimal_demos:
            self.sub_optimal_demos[key] = self.sub_optimal_demos[key][indices]
        if self.add_noise is True:
            self.adding_noise()
        
    def save_in_csv(self):
        '''
        Saves the allocations, task wise scores and sum of scores in a csv file
        '''
        rows = np.empty((self.sub_optimal_demos['X'].shape[0], (self.num_tasks*self.num_species)+(self.num_traits*self.num_species)+self.num_tasks+1))
        for i in range(self.sub_optimal_demos['X'].shape[0]):
            rows[i] = np.concatenate((self.sub_optimal_demos['X'][i].flatten(),
                                      self.sub_optimal_demos['Q'][i].flatten(),
                                    self.sub_optimal_demos['Task_scores'][i].flatten(), 
                                    self.sub_optimal_demos['Total_score'][i]))
        fields = ['X_sp1_task1', 'X_sp2_task1', 'X_sp3_task1', 'X_sp4_task1', 'X_sp1_task2', 'X_sp2_task2', 'X_sp3_task2', 'X_sp4_task2', 'X_sp1_task3', 'X_sp2_task3', 'X_sp3_task3', 'X_sp4_task3', 'Q_trait1_sp1', 'Q_trait2_sp1', 'Q_trait3_sp1', 'Q_trait1_sp2', 'Q_trait2_sp2', 'Q_trait3_sp2',  'Q_trait1_sp3', 'Q_trait2_sp3', 'Q_trait3_sp3', 'Q_trait1_sp4', 'Q_trait2_sp4', 'Q_trait3_sp4', 'score1', 'score2', 'score3', 'total_score']
        
        with open(self.csv_filename, 'w') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile) 
                
            # writing the fields 
            csvwriter.writerow(fields) 
                
            # writing the data rows 
            csvwriter.writerows(rows)

    def plot_task_scores(self):
        plt.plot(self.sub_optimal_demos['Task_scores'][:,0], norm.pdf(self.sub_optimal_demos['Task_scores'][:,0], np.mean(self.sub_optimal_demos['Task_scores'], axis=0)[0], np.std(self.sub_optimal_demos['Task_scores'], axis=0)[0]))
        plt.show()

    def print_stats(self):
        print("Max of scores for each task:", np.max(self.sub_optimal_demos['Task_scores'], axis=0))
        print("Mean of scores:", np.mean(self.sub_optimal_demos['Task_scores'], axis=0))
        print("Std deviation of scores:", np.std(self.sub_optimal_demos['Task_scores'], axis=0))

    def adding_noise(self):
        mu = np.zeros((1,3))
        print("std dev",np.std(self.sub_optimal_demos['Task_scores'], axis=0) )
        sigma = (np.std(self.sub_optimal_demos['Task_scores'], axis=0))*0.03
        self.sigma_noise = sigma
        
        noise = np.random.normal(mu, sigma, [self.sub_optimal_demos['X'].shape[0], 3])
        noise = np.reshape(noise, (self.sub_optimal_demos['X'].shape[0], 1, self.num_tasks))
        self.sub_optimal_demos['Task_scores'] = self.sub_optimal_demos['Task_scores']+noise
        self.sub_optimal_demos['Total_score'] = np.sum(self.sub_optimal_demos['Task_scores'], axis =2)
        
    def randomize_team(self):
        team_ = np.zeros(self.num_species)
        for i in range(self.num_species):
            if self.og_num_agents_per_species[i]-2 > 0:
                team_[i] = random.randint(self.og_num_agents_per_species[i]-2, self.og_num_agents_per_species[i]+2)
            else:
                team_[i] = random.randint(0, self.og_num_agents_per_species[i]+2)
        
        print("The new team is : ", team_)
        self.num_agents_per_species = team_.astype(int)
        
    def randomize_mu_var(self):
        
        mu_ = np.zeros((self.num_traits, self.num_traits))
        var_ = np.zeros((self.num_traits, self.num_traits))
        for i in range(self.num_tasks):
            for j in range(self.num_traits):
                mu_[i][j] = round(random.uniform(self.mu[i][j]-(0.25*self.mu[i][j]), self.mu[i][j]+(0.25*self.mu[i][j])), 4)     
        
        for i in range(self.num_tasks):
            for j in range(self.num_traits):
                var_[i][j] = round(random.uniform(self.var[i][j]-(0.25*self.var[i][j]), self.var[i][j]+(0.25*self.var[i][j])), 4)
        
        print("The new mu is:\n", mu_)
        print("The new var is:\n", var_)
        self.mu = mu_
        self.var = var_
        
def main():
    num_species = 4 #drone,rover,mini-rover,mini-drone
    num_tasks = 3  #move debris, search an environment, retrieve object from narrow passage
    num_traits = 3 #speed,payload,battery

    num_agents_per_species = [7, 3, 4, 6]
    csv_filename = "demos_" + time.strftime("%m%d_%H%M%S")+ ".csv"
    num_demo = 1000 #approximate number of demos that will be generated
    add_noise=True
    randomize_team=True
    randomize_mu_var=True
    demo = demos(num_species, num_agents_per_species, num_tasks, num_traits, csv_filename,
                 num_demo, add_noise=add_noise, random_team=True, random_mu_var=False)
    print("Species-Trait matrix:\n", demo.get_custom_Q())
    demo.print_stats()
    print("File name:", csv_filename)
    df = pd.read_csv(csv_filename)
    print("Number of demos: ", df.shape[0])
    df1 = df[['X_sp1_task1', 'X_sp2_task1', 'X_sp3_task1', 'X_sp4_task1', 'X_sp1_task2', 'X_sp2_task2', 'X_sp3_task2', 'X_sp4_task2', 'X_sp1_task3', 'X_sp2_task3', 'X_sp3_task3', 'X_sp4_task3', 'score1', 'score2', 'score3', 'total_score']]
    indices_del = df1.index[(df1["score1"]<0.01) | (df1["score2"]<0.01) | (df1["score3"]<0.01)].tolist()
    df1.drop(df[(df1["score1"]<0.01) | (df1["score2"]<0.01) | (df1["score3"]<0.01)].index, inplace = True)
    print("After dropping scores that are too low: ",df1.shape[0])
    demo.plot_task_scores()

if __name__ == "__main__":
    main()
