import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)
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


    
iterations = 10 ** 6
ep_decay = .99999
alpha_decay = .999999
Q = np.zeros([8, 8, 2, 5])
alpha = .9
Q_diff = []
num_iteration = []
num_iteration = []
epsilon = 1.0
playerA_ball = 0
playerB_ball = 1
game_ball = 1
gamma = 0.9
player_A_pos = 2
player_B_pos = 5
done = 0
starting_positions = [1, 2, 5, 6]

def when_done(lista):

    while True:
        s=np.random.randint(len(lista))
        k=np.random.randint(len(lista))
        if k!=s:
            break
        
    return lista[k],lista[s]

        
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

        last_q = Q[2, 1, 1, 2]
        
        if epsilon > np.random.random():
            
            player_A_action = np.random.choice(5)
            player_B_action = np.random.choice(5)
        else:
            
            player_A_action = np.argmax(Q[current_playerA, current_playerB, ball])
           # player_B_action = np.argmax(Qa[b, a, ball])
            
            player_B_action = np.random.choice(5)

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


        Q[current_playerA, current_playerB, ball, player_A_action] = (1 - alpha) * Q[current_playerA, current_playerB, ball, player_A_action] + alpha * ((1 - gamma) * (Rew_A) + gamma * np.max(Q[player_A_pos, player_B_pos, game_ball]))
        epsilon *=ep_decay
        alpha *= alpha_decay

        if i%5==0:
            
            Q_difference = np.abs(Q[2, 1, 1, 2] - last_q)
            Q_diff.append(Q_difference)
            
            num_iteration.append(i)
            print("Iteration # ", i,"alpha", alpha)




plt.plot(num_iteration,Q_diff,color='black')
plt.title('Q Learning')
plt.xlabel('Simulation Iteration')
plt.ylabel('Q Value Difference')
plt.show()