# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 11:25:10 2019

@author: safin
"""


import matplotlib.pyplot as plt
import numpy as np
import random as rand

def plot_3rd_diagram(x,y):
    
    # y=rsme_vector_for_ploting
    # x=np.array(lamda_vec)
    
    plt.figure(figsize=(10,10))
    plt.plot(x,y)
    plt.plot(x,y,'bo')
    
    plt.xlabel(r"$\lambda$", fontsize=18)
    plt.ylabel('RMSE Error', fontsize=16)
    plt.title("Figure 3")
    plt.show()
def generate_a_dataset():
    statehistory_vector =[3]
    samples_of_winnins=[]
    for k in range (100):
      
        set_of_ten = []
        
        for s in range(10):
            
            currentStates = [0,0,0,1,0,0,0]
            statehistory_vector =[3]
            for i in range(1000):
                              
                action=rand.choices(['right','left'])
                if action[0] =='left':
                                       
                    value = currentstate(currentStates) -1
                    currentStates=[0,0,0,0,0,0,0]
                    currentStates[value]=1
                    
                    
                if action[0] =='right':
                    
                    value = currentstate(currentStates) +1
                    currentStates=[0,0,0,0,0,0,0]
                    currentStates[value]=1
                    
                statehistory_vector.append(value)
                if value==0:
                    
                    break
                if value==6:
                   
                    break
            set_of_ten.append(statehistory_vector)
           
        samples_of_winnins.append(set_of_ten)
    return samples_of_winnins
def currentstate(vector):
    
    for i in range(len(vector)):
        if i== 1:
            return vector.index(i)
        
def P_t(weight,x_t):
    
    weight=np.array(weight)
    x_t=np.array(x_t)
    return np.dot(weight,x_t)
        
def into_unit_vector(value):
    if value == 0:
        print("it ended on the left")
        return 0
    if value == 6:
        print("ended on the right")
        return 1
    if value == 1:
        return np.array([1,0,0,0,0])
    if value == 2:
        return np.array([0,1,0,0,0])
    if value == 3:
        return np.array([0,0,1,0,0])
    if value == 4:
        return np.array([0,0,0,1,0])
    if value == 5:
        return np.array([0,0,0,0,1])
    
def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))

sample_dataset=generate_a_dataset()

ideal_prediction = np.array([1/6,1/3,1/2,2/3,5/6])
lamda_vec = [0,.1,.3,.5,.7,.9,1]
sum_delta_t_vec=[]
counter=0
epsilon=1
alpha=.006
avg_weight_vec=[]
rmse_to_plot_vec=[]
checkout_this_value=[]
some_i = np.array([rand.uniform(0, 1),rand.uniform(0, 1),rand.uniform(0, 1),rand.uniform(0, 1),rand.uniform(0, 1)])
print("Running first experiment with random initial weight = ",np.round(some_i,3),"and alpha = ",alpha, "\n")
for lamda in lamda_vec:    
    initial_weight=some_i
    # initial_weight = np.array([rand.uniform(0, 1),rand.uniform(0, 1),rand.uniform(0, 1),rand.uniform(0, 1),rand.uniform(0, 1)])
    weight_t=initial_weight
    sum_delta_t_vec=[]
    epsilon=1
    avg_weight_vec=[]
    counter=0
    lamda=lamda
    while epsilon >.0001:        
        for sample_set in sample_dataset:
            ##for updating weights for each 10 sequence
            # each sampleset is 10 sequence
            sum_delta_t_vec=[]
            for sequence in sample_set:
                t=0
                delta_t_vector=[]
                ## this while true iterates through the sequence computes Delta_t for a given sequence
                while True:
                    
                    sum_of_gradient_times_lamda=np.array([0,0,0,0,0])
                    for k in range(t+1):
                        sum_of_gradient_times_lamda=sum_of_gradient_times_lamda+ (lamda**(t-k))*into_unit_vector(
                                sequence[k])
                        
                    if sequence[t+1]==6:
                        p_t2=1
                    elif sequence[t+1]==0:
                        p_t2=0
                    else:
                        p_t2=P_t(initial_weight,into_unit_vector(sequence[t+1]))
                        
                    p_t1= P_t(initial_weight,into_unit_vector(sequence[t]))
                    
                    alpha_times_weights = alpha*(p_t2 - p_t1)
                    
                    delta_t_vector.append(alpha_times_weights*sum_of_gradient_times_lamda)    
                    if sequence[t+1]==6 or sequence[t+1]==0:
                        sum_delta_t_vec.append(sum(delta_t_vector))
                        delta_t_vector=[]

                        break
                    t+=1
        ##update Weight
        ## 
        
        someweight=initial_weight
        avg_weight_vec.append(someweight)
        initial_weight=initial_weight+ sum(sum_delta_t_vec)
        sum_delta_t_vec=[]
        
        ## updating epsilon
        epsilon=(abs(sum(initial_weight)-sum(someweight)))
        

        counter+=1
    print("When lambda is ",lamda, ", it converged when shown ",counter*10," sequences!")
    print("Given our lambda is ",lamda," our final weight is ",np.round(initial_weight,3)," with RMSE of ",round(rmse(initial_weight,ideal_prediction),3))
    # print(rmse(initial_weight,ideal_prediction))
    print()
    
    checkout_this_value.append(rmse(sum(avg_weight_vec)/len(avg_weight_vec),ideal_prediction))
    ## Computing RMSE
    rmse_to_plot_vec.append(rmse(initial_weight,ideal_prediction))

## Plotting the final weight vs ideal

plot_3rd_diagram(lamda_vec,checkout_this_value)


        


        







































