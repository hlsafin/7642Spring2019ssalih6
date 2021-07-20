# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 11:25:10 2019

@author: safin

"""
print("Please be patient, depending on your spec, this may take 20-30 mins wait,afterwards you'll be promoted with two different graphs, one for Figure 4, and the other for Figure 5")


import matplotlib.pyplot as plt
import numpy as np
import random as rand


def plot_3rd_diagram(x,y):
     
    plt.figure(figsize=(10,10))
    plt.plot(x,y)
    plt.plot(x,y,'bo')
    
    plt.xlabel(r"$\lambda$", fontsize=18)
    plt.ylabel('RMSE Error', fontsize=16)
    plt.title("Figure 5")
    plt.show()
    
def plot_4rd_diagram(somelist,y):
     
    plt.figure(figsize=(10,10))
    new_lamda_vec=[lamda_vec[0],lamda_vec[2],lamda_vec[-2],lamda_vec[-1]]
    new_some_list=[somelist[0],somelist[2],somelist[-2],somelist[-1]]
    for i in new_some_list:
        x=new_lamda_vec[new_some_list.index(i)]
        x=str(x)

        plt.plot(y,i,label=r"$\lambda = $"+x)
    
    plt.xlabel(r"$\alpha$", fontsize=18)
    plt.ylabel('RMSE Error', fontsize=16)
    plt.title("Figure 4")
    plt.legend()
    plt.show
         

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

mean_vec=[]
list_of_alpha_rmse=[]
list_of_list_tograph=[]
sample_dataset=generate_a_dataset()
ideal_prediction = np.array([1/6,1/3,1/2,2/3,5/6])
lamda_vec=[0,.2,.3,.4,.6,.8,1]
sum_delta_t_vec=[]
counter=0
epsilon=1
alpha_vec=[0.01,.1,.2,.3,.4,.5,.58]
avg_weight_vec=[]
rmse_to_plot_vec=[]

weight=np.array([.5,.5,.5,.5,.5])
rmse_vec=[]
for lamda in lamda_vec:
    for alpha in alpha_vec:     
        alpha=alpha

        initial_weight=weight
       
        weight_t=initial_weight
        
        sum_delta_t_vec=[]
        epsilon=1
        avg_weight_vec=[]
        counter=0    
        
        lamda=lamda
        while True:
               
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
                            someweight=initial_weight
#                               
                            initial_weight=initial_weight+ sum(sum_delta_t_vec)
                            epsilon=(abs(sum(initial_weight)-sum(someweight)))
                            sum_delta_t_vec=[]
                            delta_t_vector=[]
    
                            break
                        t+=1
            ##update Weight
          
            rmse_vec.append(rmse(initial_weight,ideal_prediction))
            initial_weight=weight
            sum_delta_t_vec=[]
            delta_t_vector=[]
            counter+=1           
            if counter==100:         
                break          
        
        mean_vec.append(sum(rmse_vec)/len(rmse_vec))
        rmse_vec=[]
        
        if sum(mean_vec)/len(mean_vec)>1:
            
            list_of_alpha_rmse.append(float('NaN'))
        else:      
            list_of_alpha_rmse.append(sum(mean_vec)/len(mean_vec))
        
        mean_vec=[]
    list_of_list_tograph.append(list_of_alpha_rmse)
    list_of_alpha_rmse=[]
    



##              Figure 5
km=list_of_list_tograph

list_of_list_tograph=km

winner_alpha_vecs=[]
for i in km:
    winner_alpha_vecs.append(min(i))

list_of_list_tograph=[]
for i in range(len(winner_alpha_vecs)):
    sample_dataset=generate_a_dataset()
    
    ideal_prediction = np.array([1/6,1/3,1/2,2/3,5/6])

    lamda_vec=[0,.2,.3,.4,.6,.8,1]

    sum_delta_t_vec=[]
    counter=0
    epsilon=1     
    avg_weight_vec=[]
    rmse_to_plot_vec=[]

    weight=np.array([.5,.5,.5,.5,.5])
    rmse_vec=[]
    mean_vec=[]          
    alpha=winner_alpha_vecs[i]            
    initial_weight=weight
    weight_t=initial_weight    
    sum_delta_t_vec=[]   
    avg_weight_vec=[]
    lamda=lamda_vec[i]
    
    while True:
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
                        someweight=initial_weight
                        ## u pdate weight vector
                        initial_weight=initial_weight+ sum(sum_delta_t_vec)
                        ##calculate RMSE
                        rmse_vec.append(rmse(initial_weight,ideal_prediction))
                        
                        sum_delta_t_vec=[]
                        delta_t_vector=[]

                        break
                    t+=1            
        counter+=1#          
        if counter==100:            
            break
            
        
    mean_vec.append(sum(rmse_vec)/len(rmse_vec))
    rmse_vec=[]

    if sum(mean_vec)/len(mean_vec)>1:
        
        list_of_alpha_rmse.append(float('NaN'))
    else:      
        list_of_alpha_rmse.append(sum(mean_vec)/len(mean_vec))
    
    mean_vec=[]


list_of_list_tograph.append(list_of_alpha_rmse)
list_of_alpha_rmse=[]


 

## graphing Figure 4
plot_4rd_diagram(km,alpha_vec)

## graph of Figure 5
plot_3rd_diagram(lamda_vec,list_of_list_tograph[0])




























