# -*- coding: utf-8 -*-
import random
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.models import load_model
import h5py
import time
import game
import util
import models
from pystockfish import *

###############################################################################
# Training
###############################################################################

if __name__ == "__main__":
    '''
    Modes
    - if with_teacher is set to True, then train with teacher
    '''
    with_teacher = True
    episodes_per_student = 25
    EPISODES = 250
    student_action_size = 1856
    start_episode = 0
    teacher_agent = models.TeacherAgent()
    student_agent = models.StudentAgent()
    teacher_agent.load('save/teacher.h5')
    #student_agent.load('save/with_random_25.h5')
    batch_size = 8
    scores_list = []
    matched_list = []
    rounds_list = []
    mat_diff_list = []

    # Creating a list of all possible actions of student agent on the chessboard
    possible_actions = util.get_possible_actions(student_action_size)

    for e in range(EPISODES):
        if e % 25 == 0: # 10 games per student
            if with_teacher:
                filename = 'save/with_teacher_' + str(int((e / 25 + 1) * 25)) + '.h5'
            else:
                filename = 'save/without_teacher_' + str(int((e / 25 + 1) * 25)) + '.h5'
            student_agent.load(filename)
            print('testing {}: {}'.format(filename, str(int((e / 25 + 1) * 25))))
        print("filename: ", filename)
        start_time = time.time()
        print_game = (e + 1) % 25 == 0
        check_mated_yet = False
        print ("episode: ", e)
        deep = Engine(depth=20) # Initialize Stockfish
        final_score = 0 # Initialize final score for the game
        done = False
        pos = game.Position(util.initial, 0, (True,True), (True,True), 0, 0)
        searcher = game.Searcher()
        moves_list = []
        round = 0
        matched = 0
        while True:
            round += 1
            if print_game:
                game.print_pos(pos)
            state = util.toBit(pos)
            before_output_list = deep.bestmove()['info'].split(" ")
            if 'mate' in before_output_list:
                if util.inCheck(pos, True):
                    print("You lost, but you're getting there little one")
                else:
                    print("Huh.  Stalemate.  ")
                end_time = time.time()
                duration = end_time - start_time
                print("episode: {}/{}, number of rounds: {}, score: {}, e: {:.2}, time : {}"
                      .format(e, EPISODES, round, final_score / float(round), student_agent.epsilon, duration / 60.0))
                scores_list.append(final_score / float(round))
                print("scores list")
                print (scores_list)
                matched_list.append(float(matched) / float(round))
                print("number of matches list")
                print (matched_list)
                print("number of rounds list")
                rounds_list.append(round)
                print(rounds_list)
                print("material differences list")
                mat_diff = util.get_material_difference(pos)
                mat_diff_list.append(mat_diff)
                print(mat_diff_list)
                done = True
                #game.print_pos(pos)
                break
            else:
                score_before_model_move = (-1)*int(before_output_list[9]) # changed from 9

            # get possible valid moves of student
            possibly_valid_moves = [m for m in pos.gen_moves(False)]
            possibly_valid_move_indices = [possible_actions.index(gm) for gm in possibly_valid_moves]

            '''Begin check for check code'''
            valid_move_indices = []
            for index in possibly_valid_move_indices:
                newPos = pos.getNewState(possible_actions[index])
                if not util.inCheck(newPos, True):
                    valid_move_indices.append(index)
            if len(valid_move_indices) == 0:
                if util.inCheck(pos, True):
                    print("You lost, but you're getting there little one")
                else:
                    print("Huh.  Stalemate.  ")
                end_time = time.time()
                duration = end_time - start_time
                print("episode: {}/{}, number of rounds: {}, score: {}, e: {:.2}, time : {}"
                      .format(e, EPISODES, round, final_score / float(round), student_agent.epsilon, duration / 60.0))
                scores_list.append(final_score / float(round))
                print("scores list")
                print (scores_list)
                matched_list.append(float(matched) / float(round))
                print("number of matches list")
                print (matched_list)
                print("number of rounds list")
                rounds_list.append(round)
                print(rounds_list)
                print("material differences list")
                mat_diff = util.get_material_difference(pos)
                mat_diff_list.append(mat_diff)
                print(mat_diff_list)
                done = True
                #game.print_pos(pos)
                break
            ''' End check for check code'''

            dqn_move_index = student_agent.act(state, valid_move_indices)
            output = deep.bestmove()
            best_move = (util.convert_to_nums(output['move'][0:2]),util.convert_to_nums(output['move'][2:]))
            if best_move[0] == None or best_move[1] == None:
                matched += 0
            else:
                best_move_index = possible_actions.index(best_move)
                if dqn_move_index == best_move_index:
                    matched += 1

            # ''' TEACHER '''
            # if with_teacher:
            #     copy_moves_list = moves_list[:]
            #     teacher_state, optimal_piece_move_indices_maybe, best_move_index, had_a_nan = util.getTeacherState(dqn_move_index, valid_move_indices, possible_actions, copy_moves_list, deep)
            #     if print_game:
            #         print('teacher state: ', teacher_state)
            #     if had_a_nan:
            #         game.print_pos(pos)
            #
            #     # get teacher action
            #     teacher_action_index = teacher_agent.act(teacher_state)
            #     if len(optimal_piece_move_indices_maybe) == 0:
            #         teacher_action_index = 0
            #         print('error avoided')
            #     if teacher_action_index == 1:
            #         move_index_based_on_partial = student_agent.act(state, optimal_piece_move_indices_maybe)
            #         while move_index_based_on_partial not in optimal_piece_move_indices_maybe:
            #              move_index_based_on_partial = random.choice(optimal_piece_move_indices_maybe)
            #         dqn_move_index = move_index_based_on_partial
            #         if print_game:
            #             print("Partial mint, chocolate chip mint")
            #         optimal_piece_moves = []
            #         for i in optimal_piece_move_indices_maybe:
            #             optimal_piece_moves.append(possible_actions[i])
            #         if print_game:
            #             print("optimal piece moves: ", optimal_piece_moves)
            #     elif teacher_action_index == 2:
            #         if print_game:
            #             print("Full hint: ", possible_actions[best_move_index])
            #         dqn_move_index = best_move_index
            #     else:
            #         if print_game:
            #             print("Not a bit of a hint (no hint)")
            #         assert teacher_action_index == 0
            #     ''' TEACHER '''

            # STUDENT ACTUALLY ACTS #
            dqn_move = possible_actions[dqn_move_index]
            # flip move
            flipped_dqn_move = util.flip_move(dqn_move)
            pos = pos.move(dqn_move, True) ## used to be new_dqn_move
            # update stockfish based on DQN action
            dqn_move_stockfish = game.render(119-flipped_dqn_move[0]) + game.render(119-flipped_dqn_move[1]) ## used to be dqn_move
            moves_list.append(dqn_move_stockfish)
            #print("dqn move stockfish: ", str(dqn_move_stockfish))
            #print(moves_list)
            deep.setposition(moves_list)


            # compute score of board after student agent makes action
            after_output = deep.bestmove()
            after_output_list = after_output['info'].split(" ")
            if 'mate' in after_output_list:
                if util.inCheck(pos, True):
                    print("You lost, but you're getting there little one")
                else:
                    print("Huh.  Stalemate.  ")
                end_time = time.time()
                duration = end_time - start_time
                print("episode: {}/{}, number of rounds: {}, score: {}, e: {:.2}, time : {}"
                      .format(e, EPISODES, round, final_score / float(round), student_agent.epsilon, duration / 60.0))
                scores_list.append(final_score / float(round))
                print("scores list")
                print (scores_list)
                matched_list.append(float(matched) / float(round))
                print("number of matches list")
                print (matched_list)
                print("number of rounds list")
                rounds_list.append(round)
                print(rounds_list)
                print("material differences list")
                mat_diff = util.get_material_difference(pos)
                mat_diff_list.append(mat_diff)
                print(mat_diff_list)
                done = True
                #game.print_pos(pos)
                # new_state = util.toBit(pos.getNewState(dqn_move))
                # student_agent.remember(state, dqn_move_index, -5000, new_state, done)
                break
            else:
                score_after_model_move = (-1)*int(after_output['info'].split(" ")[9]) # changed from 9

            # Q-Learning
            pos.rotate()
            #new_state = util.toBit(pos.getNewState(dqn_move))
            reward = score_after_model_move - score_before_model_move
            final_score += reward
            # student_agent.remember(state, dqn_move_index, reward, new_state, done)
            #state = new_state

            if print_game:
                game.print_pos(pos.rotate())
            else:
                pos.rotate()

            # ''' Teacher Q-learning '''
            # if with_teacher:
            #     if teacher_action_index != 2:
            #         score_student = get_move_value(dqn_move_index, moves_list, possible_actions, deep)
            #         optimal_move_index = possible_actions.index((convert_to_nums(after_output['move'][0:2]),convert_to_nums(after_output['move'][2:])))
            #         score_optimal = get_move_value(optimal_move_index, moves_list, possible_actions, deep)
            #         eta = 0 if teacher_index == 0 else 800
            #         reward = 1200.0 + score_student - score_optimal + eta #Use ETA if teacher_action_index = 1
            #         if len(teacher_agent.not_yet_rewarded) > 0:
            #             most_recent = teacher_agent.not_yet_rewarded[-1]
            #             if len(most_recent) == 3:
            #                 teacher_agent.remember(most_recent[0], most_recent[1], reward, teacher_state, most_recent[2])
            #             else:
            #                 assert len(most_recent) == 4
            #                 teacher_agent.remember(most_recent[0], most_recent[1], most_recent[3], teacher_state, most_recent[2])
            #             #Puts new state as last entry in the not yet remembered iteration list
            #         for not_yet_remembered_iteration in teacher_agent.not_yet_rewarded:
            #             teacher_agent.remember(not_yet_remembered_iteration[0], not_yet_remembered_iteration[1], reward, teacher_state, not_yet_remembered_iteration[2])
            #         # teacher_agent.remember(teacher_state, teacher_action_index, reward, new_teacher_state, done)
            #         teacher_agent.not_yet_remembered = [[teacher_state, teacher_action_index, done, reward]]
            #     else:
            #         not_yet_remembered_list = [teacher_state, teacher_action_index, done]
            #         teacher_agent.not_yet_rewarded.append(not_yet_remembered_list)
            # ''' Teacher Q-learning '''

            '''Begin check for check code'''
            possibly_valid_moves = [m for m in pos.gen_moves(True)]
            possibly_valid_move_indices = [possible_actions.index(gm) for gm in possibly_valid_moves]
            valid_move_indices = []
            for index in possibly_valid_move_indices:
                newPos = pos.getNewState(possible_actions[index])
                if not util.inCheck(newPos, True):
                    valid_move_indices.append(index)
            if len(valid_move_indices) == 0:
                if util.inCheck(pos, True):
                    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    print("Hahaha! We won.")
                    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                else:
                    print("Hahaha! Stalemate. ")
                end_time = time.time()
                duration = end_time - start_time
                print("episode: {}/{}, number of rounds: {}, score: {}, e: {:.2}, time : {}"
                      .format(e, EPISODES, round, final_score / float(round), student_agent.epsilon, duration / 60.0))
                scores_list.append(final_score / float(round))
                print("scores list")
                print (scores_list)
                matched_list.append(float(matched) / float(round))
                print("number of matches list")
                print (matched_list)
                print("number of rounds list")
                rounds_list.append(round)
                print(rounds_list)
                print("material differences list")
                mat_diff = util.get_material_difference(pos)
                mat_diff_list.append(mat_diff)
                print(mat_diff_list)
                done = True
                #game.print_pos(pos)
                break

            ''' End check for check code'''

            ''' if there is a problem in the future with valid moves, it might be because sunfish moves into check '''

            # Opponent takes an action
            opponent_move, score = searcher.search(pos, secs=2)
            opponent_move_stockfish = game.render(119-opponent_move[0]) + game.render(119-opponent_move[1])
            pos = pos.move(opponent_move, False)
            moves_list.append(opponent_move_stockfish)
            deep.setposition(moves_list)

            # # take care of replay
            # if len(student_agent.memory) > student_agent.batch_size:
            #     student_agent.replay(student_agent.batch_size)

            # if len(teacher_agent.memory) > batch_size:
            #     teacher_agent.replay(batch_size)

        # save teacher every name
        # if with_teacher:
        #     student_name = 'save/with_teacher_' + str(e) + '.h5'
        # else:
        #     student_name = 'save/without_teacher_' + str(e) + '.h5'
        # print("saving " + student_name)
        # student_agent.save(student_name)










# # -*- coding: utf-8 -*-
# import random
# import math
# import gym
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import deque
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam
# from keras.models import model_from_json
# from keras.models import load_model
# import h5py
#
# # from keras import models
# # from keras import layers
# # from keras.wrappers.scikit_learn import KerasClassifier
# # from sklearn.model_selection import GridSearchCV
# # from sklearn.datasets import make_classification
# import game
# from pystockfish import *
# # random.seed(3)
# # np.random.seed(3)
#
#
# ########### TEACHING AGENT #############################
#
# class TeacherAgent:
#     def __init__(self, state_size=4, action_size=3):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.memory = deque(maxlen=2000)
#         self.gamma = 0.95    # discount rate
#         self.epsilon = 0.0  # exploration rate
#         self.epsilon_min = 0.01
#         self.epsilon_decay = 0.1
#         self.learning_rate = 0.001
#         self.model = self._build_model()
#         self.moves_since_hint = 0
#         self.not_yet_rewarded = []
#
#     def _build_model(self):
#         # Neural Net for Deep-Q learning Model
#         model = Sequential()
#         model.add(Dense(24, input_dim=self.state_size, activation='relu'))
#         model.add(Dense(24, activation='relu'))
#         model.add(Dense(self.action_size, activation='linear'))
#         model.compile(loss='mse',
#                       optimizer=Adam(lr=self.learning_rate))
#         return model
#
#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))
#
#     def act(self, state):
#         if self.moves_since_hint == 2: #Meaning the teacher waits at most 3 moves for a reward, could be tweaked
#             self.moves_since_hint = 0
#             return 2 #This enforces that we don't go too long without giving hints (could change to full OR partial hint l8r)
#         if np.random.rand() <= self.epsilon:
#             random_index = random.randrange(self.action_size)
#             if random_index == 0:
#                 self.moves_since_hint += 1
#             else:
#                 self.moves_since_hint = 0
#             return random_index
#         act_values = self.model.predict(state)
#         nonrandom_index = np.argmax(act_values[0])  # returns action
#         if nonrandom_index == 0:
#             self.moves_since_hint += 1
#         else:
#             self.moves_since_hint = 0
#         return nonrandom_index
#
#     def replay(self, batch_size):
#         minibatch = random.sample(self.memory, int(batch_size))
#         for state, action, reward, next_state, done in minibatch:
#             target = reward
#             # if not done: #USED TO BE COMMENTED IN
#             #print ("teacher agent state (should be an array with shape (4, )): ", next_state)
#             # target = (reward + self.gamma *
#             #           np.amax(self.model.predict(next_state)[0]))
#             whole_list = self.model.predict(next_state)
#             amax_result = np.amax(whole_list[0])
#             target = reward + self.gamma * amax_result
#             # tabbed over part above
#
#             target_f = self.model.predict(state)
#             target_f[0][action] = target
#             self.model.fit(state, target_f, epochs=1, verbose=0)
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay
#
#     def load(self, name):
#         self.model.load_weights(name)
#
#     def save(self, name):
#         self.model.save_weights(name)
#
#
# ###############################################################################
# # Variables and constants for sunfish
# ###############################################################################
#
# piece = { 'P': 100, 'N': 280, 'B': 320, 'R': 479, 'Q': 929, 'K': 60000 }
# pst = {
#     'P': (   0,   0,   0,   0,   0,   0,   0,   0,
#             78,  83,  86,  73, 102,  82,  85,  90,
#              7,  29,  21,  44,  40,  31,  44,   7,
#            -17,  16,  -2,  15,  14,   0,  15, -13,
#            -26,   3,  10,   9,   6,   1,   0, -23,
#            -22,   9,   5, -11, -10,  -2,   3, -19,
#            -31,   8,  -7, -37, -36, -14,   3, -31,
#              0,   0,   0,   0,   0,   0,   0,   0),
#     'N': ( -66, -53, -75, -75, -10, -55, -58, -70,
#             -3,  -6, 100, -36,   4,  62,  -4, -14,
#             10,  67,   1,  74,  73,  27,  62,  -2,
#             24,  24,  45,  37,  33,  41,  25,  17,
#             -1,   5,  31,  21,  22,  35,   2,   0,
#            -18,  10,  13,  22,  18,  15,  11, -14,
#            -23, -15,   2,   0,   2,   0, -23, -20,
#            -74, -23, -26, -24, -19, -35, -22, -69),
#     'B': ( -59, -78, -82, -76, -23,-107, -37, -50,
#            -11,  20,  35, -42, -39,  31,   2, -22,
#             -9,  39, -32,  41,  52, -10,  28, -14,
#             25,  17,  20,  34,  26,  25,  15,  10,
#             13,  10,  17,  23,  17,  16,   0,   7,
#             14,  25,  24,  15,   8,  25,  20,  15,
#             19,  20,  11,   6,   7,   6,  20,  16,
#             -7,   2, -15, -12, -14, -15, -10, -10),
#     'R': (  35,  29,  33,   4,  37,  33,  56,  50,
#             55,  29,  56,  67,  55,  62,  34,  60,
#             19,  35,  28,  33,  45,  27,  25,  15,
#              0,   5,  16,  13,  18,  -4,  -9,  -6,
#            -28, -35, -16, -21, -13, -29, -46, -30,
#            -42, -28, -42, -25, -25, -35, -26, -46,
#            -53, -38, -31, -26, -29, -43, -44, -53,
#            -30, -24, -18,   5,  -2, -18, -31, -32),
#     'Q': (   6,   1,  -8,-104,  69,  24,  88,  26,
#             14,  32,  60, -10,  20,  76,  57,  24,
#             -2,  43,  32,  60,  72,  63,  43,   2,
#              1, -16,  22,  17,  25,  20, -13,  -6,
#            -14, -15,  -2,  -5,  -1, -10, -20, -22,
#            -30,  -6, -13, -11, -16, -11, -16, -27,
#            -36, -18,   0, -19, -15, -15, -21, -38,
#            -39, -30, -31, -13, -31, -36, -34, -42),
#     'K': (   4,  54,  47, -99, -99,  60,  83, -62,
#            -32,  10,  55,  56,  56,  55,  10,   3,
#            -62,  12, -57,  44, -67,  28,  37, -31,
#            -55,  50,  11,  -4, -19,  13,   0, -49,
#            -55, -43, -52, -28, -51, -47,  -8, -50,
#            -47, -42, -43, -79, -64, -32, -29, -32,
#             -4,   3, -14, -50, -57, -18,  13,   4,
#             17,  30,  -3, -14,   6,  -1,  40,  18),
# }
# # Pad tables and join piece and pst dictionaries
# for k, table in pst.items():
#     padrow = lambda row: (0,) + tuple(x+piece[k] for x in row) + (0,)
#     pst[k] = sum((padrow(table[i*8:i*8+8]) for i in range(8)), ())
#     pst[k] = (0,)*20 + pst[k] + (0,)*20
#
# # Our board is represented as a 120 character string. The padding allows for
# # fast detection of moves that don't stay within the board.
# A1, H1, A8, H8 = 91, 98, 21, 28
# initial = (
#     '         \n'  #   0 -  9
#     '         \n'  #  10 - 19
#     ' rnbqkbnr\n'  #  20 - 29
#     ' pppppppp\n'  #  30 - 39
#     ' ........\n'  #  40 - 49
#     ' ........\n'  #  50 - 59
#     ' ........\n'  #  60 - 69
#     ' ........\n'  #  70 - 79
#     ' PPPPPPPP\n'  #  80 - 89
#     ' RNBQKBNR\n'  #  90 - 99
#     '         \n'  # 100 -109
#     '         \n'  # 110 -119
# )
#
# # Lists of possible moves for each piece type.
# N, E, S, W = -10, 1, 10, -1
# directions = {
#     'P': (N, N+N, N+W, N+E),
#     'N': (N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W),
#     'B': (N+E, S+E, S+W, N+W),
#     'R': (N, E, S, W),
#     'Q': (N, E, S, W, N+E, S+E, S+W, N+W),
#     'K': (N, E, S, W, N+E, S+E, S+W, N+W)
# }
#
# # Mate value must be greater than 8*queen + 2*(rook+knight+bishop)
# # King value is set to twice this value such that if the opponent is
# # 8 queens up, but we got the king, we still exceed MATE_VALUE.
# # When a MATE is detected, we'll set the score to MATE_UPPER - plies to get there
# # E.g. Mate in 3 will be MATE_UPPER - 6
# MATE_LOWER = piece['K'] - 10*piece['Q']
# MATE_UPPER = piece['K'] + 10*piece['Q']
#
# # The table size is the maximum number of elements in the transposition table.
# TABLE_SIZE = 1e8
#
# # Constants for tuning search
# QS_LIMIT = 150
# EVAL_ROUGHNESS = 20
#
# ###############################################################################
# # DQN Model
# ###############################################################################
#
# class StudentAgent: #### NEWEST: [0.0042981037511488785, 3.0, 0.9213798872899134]
#     # new params: [0.0019518585083675654, 5.0, 0.8639910333096159]
#     def __init__(self, state_size=18, action_size=1856):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.memory = deque(maxlen=2000)
#         self.gamma = 0.9213798872899134 #0.995 #0.8639910333096159    # discount rate
#         self.epsilon = 0.0 # exploration rate
#         self.epsilon_min = 0.36
#         self.epsilon_decay = 0.999
#         self.learning_rate = 0.0042981037511488785 # 0.0019518585083675654
#         self.model = self._build_model()
#         self.batch_size = 8 # used to be 32
#
#     def _build_model(self):
#         # Neural Net for Deep-Q learning Model
#         model = Sequential()
#         model.add(Dense(24, input_dim=self.state_size, activation='relu'))
#         model.add(Dense(24, activation='relu'))
#         model.add(Dense(self.action_size, activation='linear'))
#         model.compile(loss='mse',
#                       optimizer=Adam(lr=self.learning_rate))
#         return model
#
#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))
#
#     def act(self, state, valid_move_indices):
#         #print (len(valid_move_indices))
#         if np.random.rand() <= self.epsilon:
#             random_action_index = random.choice(valid_move_indices)
#             return random_action_index
#         act_values = self.model.predict(state)
#         new_act_values = []
#         for i,val in enumerate(act_values[0]):
#             if i in valid_move_indices:
#                 new_act_values.append(val)
#             else:
#                 new_act_values.append(0.0)
#         startIndex = random.randint(0, len(new_act_values) - 1)
#         newVals = []
#         for i in range(len(new_act_values) - startIndex):
#             newVals.append(new_act_values[startIndex + i])
#         for j in range(startIndex):
#             newVals.append(new_act_values[j])
#         rotated_index = np.argmax(newVals)
#         return (rotated_index + startIndex) % len(new_act_values)
#         # return np.argmax(new_act_values)
#
#     def replay(self, batch_size):
#         minibatch = random.sample(self.memory, int(batch_size))
#         for state, action, reward, next_state, done in minibatch:
#             target = reward
#             # if not done: USED TO BE NOT COMMENTED OUT
#                 #print ("student agent state length (should be 18): ", len(next_state))
#             target = (reward + self.gamma *
#                       np.amax(self.model.predict(next_state)[0]))
#             target_f = self.model.predict(state)
#             target_f[0][action] = target
#             self.model.fit(state, target_f, epochs=1, verbose=0)
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay
#
#     def load(self, name):
#         self.model.load_weights(name)
#
#     def save(self, name):
#         self.model.save_weights(name)
#
# ###############################################################################
# # Helper functions
# ###############################################################################
# '''
# Input: move index
# Output: value of the move
# '''
#
# def get_move_value(move_index, moves_list, possible_actions):
#     before_output = deep.bestmove() # computing score of board before student agent makes action
#     before_output_list = before_output['info'].split(" ")
#     score_before_that_move = (-1)*int(before_output_list[9]) # changed from 9
#     move_tuple = possible_actions[move_index]
#     move_for_stockfish = game.render(119-move_tuple[0]) + game.render(119-move_tuple[1])
#     moves_list.append(move_for_stockfish)
#     deep.setposition(moves_list) #SYNTAX!!
#     after_output = deep.bestmove()
#     after_output_list = after_output['info'].split(" ")
#     score_after_that_move = (-1)*int(after_output_list[9]) # changed from 9
#     last_thing = moves_list.pop()
#     deep.setposition(moves_list) #Does set and reset deep position -- probably not buggy, but look into
#     return score_after_that_move - score_before_that_move
#
# '''
# Convert "e2" to "85", that kind of thing
# '''
# def convert_to_nums(algebra_str):
#     #print("algebra str: ", algebra_str)
#     letter = algebra_str[0]
#     number = algebra_str[1]
#     letter_as_num = ord(letter) - ord("a") + 1
#     new_number = 10 - int(number)
#     #print("converted: ", 10 * new_number + letter_as_num)
#     return 10 * new_number + letter_as_num
# '''
# Get teacher state
# '''
# def getTeacherState(suggested_move_index, valid_move_indices, possible_actions, moves_list): #UPDATE PARAMS IN MAIN!!!
#     had_a_nan_in_teacher_state = False
#     state = []
#     # state 1: difference between suggested move value and optimal move value
#     output = deep.bestmove() ## something else
#     #print("optimal move now: ", output['move'])
#     best_move = (convert_to_nums(output['move'][0:2]),convert_to_nums(output['move'][2:]))
#     best_move_index = possible_actions.index(best_move)
#     suggested_move_value = get_move_value(suggested_move_index, moves_list, possible_actions)
#     optimal_move_value = get_move_value(best_move_index, moves_list, possible_actions)
#     diff = suggested_move_value - optimal_move_value
#     state.append(diff)
#     # state 2: optimal move
#     valid_moves_values = []# get value of move for move in possibly_valid_move_indices
#     for valid_move_index in valid_move_indices:
#         valid_moves_values.append(get_move_value(valid_move_index, moves_list, possible_actions))
#     state.append(np.std(np.array(valid_moves_values)))
#     # state 3: boolean comparing suggested and optimal
#     state.append(best_move_index == suggested_move_index)
#     # state 4: std of optimal piece moves
#     optimal_piece_move_indices = []
#     #piece_to_move_loc = best_move[0:2]# in the form of
#     reformatted_loc = best_move[0]# e2 --> 85
#     print ("whether best move is in valid move indices", best_move_index in valid_move_indices)
#     for move_index in valid_move_indices:
#         if possible_actions[move_index][0] == reformatted_loc:
#             optimal_piece_move_indices.append(move_index)
#     if len(optimal_piece_move_indices) == 0:
#         print ("length of optimal piece move indices: ", 0)
#         print (valid_move_indices)
#     print (optimal_piece_move_indices)
#     optimal_piece_move_values = []
#     for optimal_piece_move_index in optimal_piece_move_indices:
#         optimal_piece_move_values.append(get_move_value(optimal_piece_move_index, moves_list, possible_actions))
#     if len(optimal_piece_move_values) == 0:
#         had_a_nan_in_teacher_state = True
#     state.append(np.std(np.array(optimal_piece_move_values)))
#     # state 5: number of moves since partial hint or no hint
#     #state.append(teacher_agent.moves_since_hint)
#     if np.isnan(state[3]):
#         had_a_nan_in_teacher_state = True
#     state = np.reshape(state, (1,4))
#     #print('teacher state: ', state)
#     return state, optimal_piece_move_indices, best_move_index, had_a_nan_in_teacher_state
#
#
# '''
# Converts the 64-character representation of the board state used in
# game.py to a 64-bit representation.
# '''
#
# def toBit(pos):
#     board = pos.board
#     wc = pos.wc
#     bc = pos.bc
#     ep = pos.ep
#     kp = pos.kp
#     state = [0] * 18
#     indices = ['P', 'R', 'N', 'B', 'Q', 'K', 'p', 'r', 'n', 'b', 'q', 'k']
#     rows = "".join(board.split('\n')[:-1]) ## might be a bug
#     for i in range(len(rows)):
#         piece = rows[i]
#         for j, letter in enumerate(indices):
#             if piece == letter:
#                 state[j] += 2 ** i
#     wc_1, wc_2 = wc
#     bc_1, bc_2 = bc
#     state[12] = int(wc_1)
#     state[13] = int(wc_2)
#     state[14] = int(bc_1)
#     state[15] = int(bc_2)
#     state[16] = int(ep)
#     state[17] = int(kp)
#     state = np.reshape(state, (1,18))
#     return state
#
# '''
# Retrieves the king's position at any given board state.
# '''
#
# def getKingPos(boardString, piece_letter):
#     #print(piece_letter)
#     # boardString_list = []
#     # for row in position.board.split("\n"):
#     #     if row != "" and not row.isspace():
#     #         boardString_list.append(row)
#     # boardString = "".join(boardString_list)#position.board.split("\n")[3:11])
#     # #print("length of board string before reducing empty spaces: ", str(len(boardString)))
#     # boardString = boardString.replace(" ", "")
#     #print(splitted)
#     if len(boardString) != 64:
#         print("boardString length = " + str(len(boardString)))
#     #print ("boardString: ", boardString)
#     splitted = []
#     for i in range(8):
#         splitted.append(boardString[8 * i : 8 * i + 8])
#     for l, line in enumerate(splitted): # "".join(board.split('\n')[:-1])
#         #print(line)
#         for col in range(8):
#             #print (line[col])
#             if line[col] == piece_letter:
#                 return (l, col)
#     return ("nope", "nope")
#
# '''
# Checks if the input player is in check.
# '''
# def inCheck(position, checkCurrentPlayer):
#     pieceLetters = ["P", "N", "B", "R", "Q", "K"]
#     king = 'k'
#     if checkCurrentPlayer:
#         pieceLetters = ["p", "n", "b", "r", "q", "k"] ## case is same for opponent and king
#         king = 'K'
#     #print(king)
#     # pieceLetters = ["p", "n", "b", "r", "q", "k"]
#     # if not checkCurrentPlayer:
#     #     pieceLetters = ["P", "N", "B", "R", "Q", "K"] ## case is same for opponent and king
#      ## changed from pieceLetters[-1] # it's when it's lowercase that getKingPos error happens
#     #print(position.board.split("\n"))
#     boardString_list = []
#     for row in position.board.split("\n"):
#         if row != "" and not row.isspace():
#             boardString_list.append(row)
#     boardString = "".join(boardString_list)#position.board.split("\n")[3:11])
#     #print("length of board string before reducing empty spaces: ", str(len(boardString)))
#     boardString = boardString.replace(" ", "")
#     #print("length of board string: ", str(len(boardString)))
#     kRow, kCol = getKingPos(boardString, king)
#     if kRow == "nope": ## somewhat big change
#         return True
#     for row in range(8):
#         for col in range(8):
#             pieceIndex = 8 * row + col
#             piece = boardString[pieceIndex]
#             if piece == pieceLetters[0]:
#                 if row + 1 == kRow and abs(col - kCol) == 1:
#                     # print(pieceLetters[0])
#                     return True
#             if piece == pieceLetters[1]:
#                 if abs(row - kRow) == 2 and abs(col - kCol) == 1:
#                     # print(pieceLetters[1])
#                     return True
#                 if abs(row - kRow) == 1 and abs(col - kCol) == 2:
#                     # print(pieceLetters[1])
#                     return True
#             if piece == pieceLetters[2] or piece == pieceLetters[4]:
#                 if abs(row - kRow) == abs(col - kCol):
#                     canCheck = True
#                     if (row - kRow) * (col - kCol) > 0:
#                         start = 8 * min(row, kRow) + min(col, kCol)
#                         for diagonAlley in range(abs(row - kRow) - 1):
#                             # if boardString[start + 9 * diagonAlley] != ".":
#                             #     canCheck = False
#                             if boardString[start + 9 * (diagonAlley + 1)] != ".":
#                                 canCheck = False
#                         if canCheck == True:
#                             # print(piece)
#                             return True
#                     else:
#                         start = 8 * min(row, kRow) + max(col, kCol)
#                         for diagonAlley in range(abs(row - kRow) - 1):
#                             if boardString[start + 7 * (diagonAlley + 1)] != ".":
#                                 canCheck = False
#                         if canCheck == True:
#                             # print(piece)
#                             return True
#             if piece == pieceLetters[3] or piece == pieceLetters[4]:
#                 if row == kRow:
#                     canCheck = True
#                     for inBetween in range(min(col, kCol) + 1, max(col, kCol)):
#                         if boardString[8 * row + inBetween] != ".":
#                             canCheck = False
#                     if canCheck == True:
#                         # print(piece)
#                         return True
#                 if col == kCol:
#                     canCheck = True
#                     for inBetween in range(min(row, kRow) + 1, max(row, kRow)):
#                         if boardString[8 * inBetween + col] != ".":
#                             canCheck = False
#                     if canCheck == True:
#                         # print(piece)
#                         return True
#             if piece == pieceLetters[5]:
#                 if abs(row - kRow) <= 1 and abs(col - kCol) <= 1:
#                     # print(pieceLetters[5])
#                     return True
#     return False
#
#
# ###############################################################################
# # Training
# ###############################################################################
# ##### TRAINING STUDENT WITOUT TEACHER NOW STARTING 11:25am #####
#
# if __name__ == "__main__": #def get_four_game_average_score(student_agent):
#     # Constants for training
#     EPISODES = 6
#     student_action_size = 1856
#
#     for i in [100]:
#         if i % 100 == 0:
#             student_agent = StudentAgent()
#             filename = 'save/with_teacher_' + str(i) + '.h5'
#             student_agent.load(filename)
#
#             # Initialize agents
#             #student_agent = StudentAgent()
#             teacher_agent = TeacherAgent()
#             batch_size = 8  # changed from 32
#
#             # Creating a list of all possible actions of student agent on the chessboard
#             possible_actions = []
#             for x_prev in range(2,10):
#                 for y_prev in range(1,9):
#                     for x_next in range(2,10):
#                         for y_next in range(1,9):
#                             if x_next == x_prev or y_next == y_prev or abs(x_prev - x_next) == abs(y_prev - y_next):
#                                 possible_actions.append((int(str(x_prev) + str(y_prev)), int(str(x_next) + str(y_next))))
#                             elif abs(x_prev - x_next) <= 1 and abs(y_prev - y_next) <= 1:
#                                 possible_actions.append((int(str(x_prev) + str(y_prev)), int(str(x_next) + str(y_next))))
#                             elif abs(x_prev - x_next) == 1 and abs(y_prev - y_next) == 2:
#                                 possible_actions.append((int(str(x_prev) + str(y_prev)), int(str(x_next) + str(y_next))))
#                             elif abs(x_prev - x_next) == 2 and abs(y_prev - y_next) == 1:
#                                 possible_actions.append((int(str(x_prev) + str(y_prev)), int(str(x_next) + str(y_next))))
#             assert len(possible_actions) == student_action_size
#
#
#             # TEACHER ACTION LIST
#             #['no_hint', 'partial_hint', 'full_hint']
#
#             scores_list = []
#
#             # Training EPISODES times
#             for e in range(EPISODES):
#                 print_game = (e + 1) % 25 == 0
#                 check_mated_yet = False
#                 print ("episode: ", e)
#                 deep = Engine(depth=20) # Initialize Stockfish
#                 final_score = 0 # Initialize final score for the game
#                 done = False
#                 pos = game.Position(initial, 0, (True,True), (True,True), 0, 0)
#                 searcher = game.Searcher()
#                 moves_list = []
#                 round = 0
#                 while True:
#                     round += 1
#                     if print_game:
#                         game.print_pos(pos)
#                     state = toBit(pos)
#                     before_output = deep.bestmove() # computing score of board before student agent makes action
#                     before_output_list = before_output['info'].split(" ")
#                     if 'mate' in before_output_list:
#                         mate_index = before_output_list.index('mate')
#                         mate_score = before_output_list[mate_index + 1]
#                         # if not check_mated_yet:
#                         #     score_before_model_move = -5000 #/ float(math.sqrt(float(mate_score)))
#                         #     check_mated_yet = True
#                         # else:
#                         score_before_model_move = 0
#                         print("mate is in before_output_list so score is ", str(score_before_model_move))
#                     else:
#                         score_before_model_move = (-1)*int(before_output['info'].split(" ")[9]) # changed from 9
#
#                     # get possible valid moves of student
#                     possibly_valid_moves = [m for m in pos.gen_moves(False)]
#                     possibly_valid_move_indices = [possible_actions.index(gm) for gm in possibly_valid_moves]
#                     meg_is_so_so_so_so_special = True
#                     king_is_in_check = inCheck(pos, True) #ADDED
#                     if king_is_in_check:
#                         print("********CHECK**********")
#                     '''Begin check for check code'''
#                     valid_move_indices = []
#                     for index in possibly_valid_move_indices:
#                         newPos = pos.getNewState(possible_actions[index])
#                         if not inCheck(newPos, True):
#                             valid_move_indices.append(index)
#                     if len(valid_move_indices) == 0:
#                         if inCheck(pos, True):
#                             print("You lost, but you're getting there little one")
#                             #game.print_pos(pos) #CHANGEDD think this could be a good way to tell whether game goes exactly the same way every time
#                         else:
#                             print("Huh.  Stalemate.  ")
#                         print("episode: {}/{}, number of rounds: {}, score: {}, e: {:.2}"
#                               .format(e, EPISODES, round, final_score / float(round), student_agent.epsilon))
#                         scores_list.append(final_score / float(round))
#                         print (scores_list)
#                         done = True
#                         break
#                     # else:
#                     #     valid_move_indices = possibly_valid_move_indices
#
#                     ''' End check for check code'''
#
#                     # DQN chooses an action
#                     #valid_moves = [m for m in pos.gen_moves()] #### (85,65), (87, 97)
#                     #valid_move_indices = [possible_actions.index(gm) for gm in valid_moves]
#                     dqn_move_index = student_agent.act(state, valid_move_indices)
#
#                     # # a check that we should be removing later
#                     # while dqn_move_index not in valid_move_indices:
#                     #     dqn_move_index = random.choice(valid_move_indices)
#                     #
#                     # if dqn_move_index not in valid_move_indices:
#                     #     print("made invalid move")
#                     #     break
#
#                     #suggested_dqn_move_stockfish = game.render(119-possible_actions[dqn_move_index][0]) + game.render(119-possible_actions[dqn_move_index][1]) ## used to be dqn_move
#                     #moves_list.append(suggested_dqn_move_stockfish)
#                     #print("dqn move stockfish: ", str(dqn_move_stockfish))
#                     #print(moves_list)
#
#                     ''' TEACHER '''
#                     # # get teacher state given the student's suggested move index and the state of the game
#                     # copy_moves_list = moves_list[:]
#                     # teacher_state, optimal_piece_move_indices_maybe, best_move_index, had_a_nan = getTeacherState(dqn_move_index, valid_move_indices, possible_actions, copy_moves_list)
#                     # print('teacher state: ', teacher_state)
#                     # # if had_a_nan:
#                     # #     print("Should print board")
#                     # #     game.print_pos(pos)
#                     # #moves_list.pop() # remove suggested move from moves list
#                     #
#                     # # get teacher action
#                     # teacher_action_index = teacher_agent.act(teacher_state) ## add this to teacher agent class
#                     # if teacher_action_index == 1:
#                     #     move_index_based_on_partial = student_agent.act(state, optimal_piece_move_indices_maybe)
#                     #     while move_index_based_on_partial not in optimal_piece_move_indices_maybe:
#                     #          move_index_based_on_partial = random.choice(optimal_piece_move_indices_maybe)
#                     #     dqn_move_index = move_index_based_on_partial
#                     #     print("Partial mint, chocolate chip mint")
#                     #     optimal_piece_moves = []
#                     #     for i in optimal_piece_move_indices_maybe:
#                     #         optimal_piece_moves.append(possible_actions[i])
#                     #     print("optimal piece moves: ", optimal_piece_moves)
#                     #     # partial hint was given, check which hint that was
#                     #     #If there's a BUGG: did we accidentally over-rotate board here?
#                     # elif teacher_action_index == 2:
#                     #     print("Hole hint")
#                     #     print("Full hint: ", possible_actions[best_move_index])
#                     #     dqn_move_index = best_move_index
#                     #     # full hint was given, proceed with that move
#                     # else:
#                     #     print("Not a bit of a hint (no hint)")
#                     #     assert teacher_action_index == 0
#                     #     # no hint was given, proceed with student's suggested move
#                     ''' TEACHER '''
#
#                     # STUDENT ACTUALLY ACTS #
#
#                     dqn_move = possible_actions[dqn_move_index]
#                     # flip move
#                     firstPos = dqn_move[0]
#                     firstPosRow = int(str(firstPos)[0])
#                     firstPosCol = int(str(firstPos)[1])
#                     secondPos = dqn_move[1]
#                     secondPosRow = int(str(secondPos)[0])
#                     secondPosCol = int(str(secondPos)[1])
#                     newFirstPosRow = 11 - firstPosRow
#                     newFirstPosCol = 9 - firstPosCol
#                     newSecondPosRow = 11 - secondPosRow
#                     newSecondPosCol = 9 -secondPosCol
#                     new_dqn_move = (int(str(newFirstPosRow) + str(newFirstPosCol)), int(str(newSecondPosRow)+str(newSecondPosCol)))
#                     pos = pos.move(dqn_move, True) ## used to be new_dqn_move
#                     # update stockfish based on DQN action
#                     dqn_move_stockfish = game.render(119-new_dqn_move[0]) + game.render(119-new_dqn_move[1]) ## used to be dqn_move
#                     if king_is_in_check:
#                         print("Chose to move " + dqn_move_stockfish + "to escape")
#                     moves_list.append(dqn_move_stockfish)
#                     #print("dqn move stockfish: ", str(dqn_move_stockfish))
#                     #print(moves_list)
#                     deep.setposition(moves_list)
#
#
#                     # compute score of board after student agent makes action
#                     after_output = deep.bestmove()
#                     #print ("after output info: " + after_output['info'])
#                     after_output_list = after_output['info'].split(" ")
#                     if 'mate' in after_output_list:
#                         mate_index = after_output_list.index('mate')
#                         mate_score = after_output_list[mate_index + 1]
#                         if not check_mated_yet:
#                             score_after_model_move = -5000 #/ float(math.sqrt(float(mate_score)))
#                             check_mated_yet = True
#                         else:
#                             score_after_model_move = 0
#                         print("mate is in after_output_list so score is ", str(score_after_model_move))
#                     else:
#                         score_after_model_move = (-1)*int(after_output['info'].split(" ")[9]) # changed from 9
#
#                     # Q-Learning
#                     pos.rotate()
#                     new_state = toBit(pos.getNewState(dqn_move))
#                     #print ("reward: ", score_after_model_move, " - ", score_before_model_move)
#                     reward = score_after_model_move - score_before_model_move #abs(score_after_model_move - score_before_model_move)*(score_after_model_move - score_before_model_move)
#                     #print(reward)
#                     final_score += reward
#                     #student_agent.remember(state, dqn_move_index, reward, new_state, done)
#                     state = new_state
#
#                     if print_game:
#                         game.print_pos(pos.rotate())#pos.rotate()#game.print_pos()
#                     else:
#                         pos.rotate()
#
#
#                     ''' Teacher Q-learning '''
#                     # if teacher_action_index != 2:
#                     #     score_student = get_move_value(dqn_move_index, moves_list, possible_actions)
#                     #     optimal_move_index = possible_actions.index((convert_to_nums(after_output['move'][0:2]),convert_to_nums(after_output['move'][2:])))
#                     #     score_optimal = get_move_value(optimal_move_index, moves_list, possible_actions)
#                     #     reward = 300.0 + score_student - score_optimal #Use ETA if teacher_action_index = 1
#                     #     if len(teacher_agent.not_yet_rewarded) > 0:
#                     #         most_recent = teacher_agent.not_yet_rewarded[-1]
#                     #         if len(most_recent) == 3:
#                     #             teacher_agent.remember(most_recent[0], most_recent[1], reward, teacher_state, most_recent[2])
#                     #         else:
#                     #             assert len(most_recent) == 4
#                     #             teacher_agent.remember(most_recent[0], most_recent[1], most_recent[3], teacher_state, most_recent[2])
#                     #         #Puts new state as last entry in the not yet remembered iteration list
#                     #     for not_yet_remembered_iteration in teacher_agent.not_yet_rewarded:
#                     #         teacher_agent.remember(not_yet_remembered_iteration[0], not_yet_remembered_iteration[1], reward, teacher_state, not_yet_remembered_iteration[2])
#                     #     # teacher_agent.remember(teacher_state, teacher_action_index, reward, new_teacher_state, done)
#                     #     teacher_agent.not_yet_remembered = [[teacher_state, teacher_action_index, done, reward]]
#                     # else:
#                     #     not_yet_remembered_list = [teacher_state, teacher_action_index, done]
#                     #     teacher_agent.not_yet_rewarded.append(not_yet_remembered_list)
#                     ''' Teacher Q-learning '''
#
#
#
#
#
#                     # # if checkmate happened, break out of the loop and print stats
#                     # if done:
#                     #     print("episode: {}/{}, score: {}, e: {:.2}"
#                     #           .format(e, EPISODES, round, student_agent.epsilon))
#                     #     break
#
#                     '''Begin check for check code''' # im tired!
#
#                     possibly_valid_moves = [m for m in pos.gen_moves(True)]
#                     #print(possibly_valid_moves)
#                     possibly_valid_move_indices = [possible_actions.index(gm) for gm in possibly_valid_moves]
#                     valid_move_indices = []
#                     #print("possibly valid move indices: ", str(len(possibly_valid_move_indices)))
#                     for index in possibly_valid_move_indices:
#                         newPos = pos.getNewState(possible_actions[index])
#                         #print(newPos.board)
#                         if not inCheck(newPos, True):
#                             valid_move_indices.append(index)
#                     if len(valid_move_indices) == 0:
#                         if inCheck(pos, True):
#                             print("Hahaha! We won.")
#                         else:
#                             print("Hahaha! Stalemate. ")
#                         print("episode: {}/{}, number of rounds: {}, score: {}, e: {:.2}"
#                               .format(e, EPISODES, round, final_score / float(round), student_agent.epsilon))
#                         scores_list.append(final_score / float(round))
#                         print (scores_list)
#                         done = True
#                         # if e % 10 == 0:
#                         #     score_list.append(final_score)
#                         #     current_mean = sum(scores_list) / 10.0
#                         #     if current_mean < previous_mean:
#                         #         raise Exception
#                         break
#                     # else:
#                     #     valid_move_indices = possibly_valid_move_indices
#
#                     ''' End check for check code'''
#
#                     ''' if there is a problem in the future with valid moves, it might be because sunfish moves into check '''
#
#                     #game.print_pos(pos)
#
#                     # Opponent takes an action
#                     opponent_move, score = searcher.search(pos, secs=2)
#                     opponent_move_stockfish = game.render(119-opponent_move[0]) + game.render(119-opponent_move[1])
#                     pos = pos.move(opponent_move, False)
#                     # update stockfish based on opponent action
#                     moves_list.append(opponent_move_stockfish)
#                     deep.setposition(moves_list)
#
#                     # take care of replay
#                     #if len(student_agent.memory) > student_agent.batch_size:
#                     #    student_agent.replay(student_agent.batch_size)
#
#                     # if len(teacher_agent.memory) > batch_size:
#                     #     teacher_agent.replay(batch_size)
#                 # if e % 25 == 0:
#                 #     filename = 'save/without_teacher_' + str(e) + '.h5'
#                 #     print("saving" + filename)
#                 #     student_agent.save(filename)
#
#             #return sum(scores_list) / (EPISODES + 0.0)
#             # plt.plot(scores_list)
#             # plt.ylabel('average score value')
#             # plt.xlabel('games')
#             # plt.show()
#             #student_json_string = student_agent.to_json()
#             #student_agent.save_weights(filepath)
#             #model = model_from_json(json_string)
#
#                 #     ########## OLD CODE ###########
#                 # # if e % 10 == 0:
#                 # #     agent.save("/save/cartpole-dqn.h5")
#
#         #TODOO:
#         #MAKE init_params completed
#         #MAKE ranges completed
#         #Make maxes completed
#         #MAKE mins completed
#         #MAKE assign_params pretty much completed (just need to make batch size a class var)
#         #MAKE get_four_game_average_score Not done: involves calling the original main function for 4 iterations
#         #parameter order: learning_rate, batch_size, gamma
#         # if __name__ == "__main__":
#         #     student_agent = StudentAgent()
#         #     def assign_params(params):
#         #         student_agent.learning_rate = params[0]
#         #         student_agent.batch_size = 2 ** int(params[1]) #Need to add batch size to studentAgent class
#         #         student_agent.gamma = params[2]
#         #
#         #
#         #     i_love_go_main_muse = True
#         #     ranges = [0.004, 5, 0.2] #Use this so that we can choose powers of two as batch sizes
#         #     mins = [0.001, 1, 0.8]
#         #     maxes = [0.005, 6, 1.0]
#         #     def init_params():
#         #         init_parameters = []
#         #         for i in range(3): #3 = num parameters, lr, bs, gamma
#         #             init_parameters.append(random.random() * ranges[i] + mins[i])
#         #         init_parameters[1] = random.randint(1, 6) # Set batch size discretely
#         #         return init_parameters
#         #     params = init_params()
#         #     best_average_score = -2000
#         #     num_tries = 0
#         #     while best_average_score < -200 and num_tries < 50:
#         #         print("parameters:")
#         #         print(params)
#         #         num_tries += 1
#         #         for i in range(len(params)):
#         #             print("Exploring parameter ", str(i), "with value ", str(params[i]))
#         #             increase_score = -2000
#         #             if i == 1:
#         #                 higher_val = params[i] + 1
#         #             else:
#         #                 higher_val = params[i] + 0.05 * ranges[i]
#         #             if higher_val <= maxes[i]:
#         #                 print("Increasing from " + str(params[i]) + " to " + str(higher_val))
#         #                 params[i] = higher_val
#         #                 assign_params(params)
#         #                 increase_score = get_four_game_average_score(student_agent)
#         #                 params[i] -= 0.05 * ranges[i]
#         #             decrease_score = -2000
#         #             if i == 1:
#         #                 lower_val = params[i] - 1
#         #             else:
#         #                 lower_val = params[i] - 0.05 * ranges[i]
#         #             if lower_val >= mins[i]:
#         #                 print("Decreasing from " + str(params[i]) + " to " + str(lower_val))
#         #                 params[i] = lower_val
#         #                 assign_params(params)
#         #                 decrease_score = get_four_game_average_score(student_agent)
#         #                 params[i] += 0.05 * ranges[i]
#         #             print("Old best score: ", str(best_average_score))
#         #             print("Increase score: ", str(increase_score))
#         #             print("Decrease score: ", str(decrease_score))
#         #             if max(increase_score, decrease_score) > best_average_score:
#         #                 if increase_score > decrease_score:
#         #                     best_average_score = increase_score
#         #                     params[i] += 0.05 * ranges[i]
#         #                 else:
#         #                     best_average_score = decrease_score
#         #                     params[i] -= 0.05 * ranges[i]
#
#
#         # student_agent = StudentAgent()
#         # teacher_agent = TeacherAgent()
#         # student_net = KerasClassifier(build_fn=student_agent._build_model, verbose=0)
#         # learning_rates = [0.001, 0.002, 0.003, 0.004, 0.005]
#         # batch_sizes = [2, 4, 8, 16, 32]
#         # gammas = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95]
#         # hyperparameters = dict(sgd_learning_rate = learning_rates, sgd_batch_size = batch_sizes, sgd_gamma = gammas)
#         # grid = GridSearchCV(estimator=student_net, param_grid=hyperparameters)
#         # grid_result = grid.fit(features, target)
#         # best_params = grid_result.best_params_
