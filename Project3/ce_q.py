import numpy as np
import matplotlib.pyplot as plt
#import time
from cvxopt import matrix, solvers
from numpy import  eye, hstack, ones, vstack, zeros
np.random.seed(100)
def when_done(lista):

    while True:
        s=np.random.randint(len(lista))
        k=np.random.randint(len(lista))
        if k!=s:
            break
        
    return lista[k],lista[s]
def step_temp1(action):
    player_A_action = action
    
    
    if player_A_action == 0 and player_A_pos > 3:
        player_state = player_A_pos - 4

    elif player_A_action == 1 and player_A_pos not in [3, 7]:
        player_state = player_A_pos + 1

    elif player_A_action == 2 and player_A_pos < 4:
        player_state = player_A_pos + 4

    elif player_A_action == 3 and player_A_pos not in [0, 4]:
        player_state = player_A_pos - 1

    else:
        player_state = player_A_pos
    return player_state

def step_temp2(action):
    player_B_action = action
    if player_B_action == 0 and player_B_pos > 3:
        player_state = player_B_pos - 4

    elif player_B_action == 1 and player_B_pos not in [3, 7]:
        player_state = player_B_pos + 1

    elif player_B_action == 2 and player_B_pos < 4:
        player_state = player_B_pos + 4

    elif player_B_action == 3 and player_B_pos not in [0, 4]:
        player_state = player_B_pos - 1

    else:
        player_state = player_B_pos
    return player_state

def solve_ceq(q1, q2):


    A = np.zeros((40, 25))
    row = 0

    for i in range(5):
        for j in range(5):
            if i != j:
                A[row, i * 5:(i + 1) * 5] = q1[i] - q1[j]
                A[row + 20, i:25:5] = q2[:, i] - q2[:, j]
                row += 1


    eye_mat = np.hstack((zeros((25, 1)), -eye(25)))
    A = np.vstack((np.hstack((ones((40, 1)), matrix(A))), eye_mat))
    A = matrix(np.vstack((A, np.hstack((0,ones(25))), np.hstack((0,-ones(25))))))
    b = matrix(np.hstack((zeros(65), [1, -1])))
    c = matrix(np.hstack(([-1.], -(q1+q2).flatten())))
    sol = solvers.lp(c,A,b, solver=glpksolver)
    if sol['x'] is None:
        return 0, 0


    return np.matmul(q1.flatten(), sol['x'][1:])[0]

glpksolver = 'glpk'
solvers.options['glpk'] = {'LPX_K_MSGLEV': 0, 'msg_lev': "GLP_MSG_OFF"}

playerA_ball = 0
playerB_ball = 1
game_ball = 1
iterations = 10 ** 6
alpha = 1.0   
gamma = 0.9
Qa = np.zeros([8,8,2,5,5])
Qb = np.zeros([8,8,2,5,5])
player_A_pos = 2
player_B_pos = 5
done = 0
alpha_decay = 0.99999
arbitrary = [2, 1, 1, 4, 2]
starting_positions = [1, 2, 5, 6]
Q_diff = []
num_iteration = []


for i in range(iterations):
        if done == 1:
            
            player_A_pos,player_B_pos = when_done(starting_positions)

            if np.random.randint(2) == 0:
                ball = playerA_ball
                ball_pos = player_A_pos
            else:
                ball = playerB_ball
                ball_pos = player_B_pos

            
            done = 0

        current_playerA = player_A_pos
        current_playerB = player_B_pos
        ball = game_ball

        last_q = Qa[2][1][1][4][2]

        player_A_action = np.random.randint(5)
        player_B_action = np.random.randint(5)

        Qa_state = Qa[player_A_pos, player_B_pos, game_ball]
        Qb_state = Qb[player_A_pos, player_B_pos, game_ball]
        


        if np.random.randint(2) == 0:
                  
            temp_state1= step_temp1(player_A_action)
            temp_state2 = step_temp2(player_B_action)
    

            if temp_state1 != player_B_pos:

                player_A_pos = temp_state1
            else:

                ball = playerB_ball
    
            if temp_state2 != player_A_pos:
                player_B_pos = temp_state2
            else:
                ball = playerA_ball

            if ball:
                ball_pos = player_A_pos
            else:
                ball_pos = player_B_pos
        
        else:


            temp_state1 = step_temp1(player_B_action)
            temp_state2 = step_temp2(player_A_action)

            if temp_state1 != player_B_pos:

                player_A_pos = temp_state1
            else:

                ball = playerB_ball
    
            if temp_state2 != player_A_pos:
                player_B_pos = temp_state2
            else:
                ball = playerA_ball

            if ball:
                ball_pos = player_A_pos
            else:
                ball_pos = player_B_pos


        if ball_pos in [0, 4]:
            Rew_A = 100
            Rew_B = -100
            done = 1

        elif ball_pos in [3, 7]:
            Rew_A = -100
            Rew_B = 100
            done = 1

        else:
            Rew_A = 0
            Rew_B = 0
            done = 0


        Qa[current_playerA, current_playerB, ball, player_A_action, player_B_action] = (1 - alpha) * Qa[current_playerA, current_playerB, ball, player_A_action, player_B_action] + alpha * ((1 - gamma) * Rew_A + gamma * solve_ceq(Qa_state, Qb_state))
        alpha *= alpha_decay

        Q_difference = np.abs(Qa[2, 1, 1, 2, 4] - last_q)
        Q_diff.append(Q_difference)
       
        num_iteration.append(i)
        if i%1000==0:
            print("Iteration # ", i,"alpha: ", alpha)
        


plt.plot(num_iteration,Q_diff,color='black')
plt.title('Correlated Equilibirum Q Learning')
plt.xlabel('Simulation Iteration')
plt.ylabel('Q Value Difference')
plt.show()








    
    