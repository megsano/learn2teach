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
# helper
###############################################################################
def get_four_game_average_score(teacher_agent, student_agent, tune_teacher):
    '''
    Modes
    - if with_teacher is set to True, then train with teacher
    '''
    lower_student_num = 125
    upper_student_num = 225
    if tune_teacher:
        filename = "save/without_teacher_" + str(random.randint(lower_student_num, upper_student_num)) + ".h5"
        student_agent = models.StudentAgent()
        student_agent.load(filename)
        student_list = [student_agent]
    else:
        student_file_names = ["save/without_teacher_" + str(random.randint(lower_student_num, upper_student_num)) + ".h5" for i in range(4)]
        student_list = []
        for s in student_file_names:
            student_agent = models.StudentAgent()
            student_agent.load(s)
            student_list.append(student_agent)
        #student_list = [student_agent.load(s) for s in student_file_names]
        assert len(student_list) == 4
    EPISODES = 4
    student_action_size = 1856
    batch_size = 8
    scores_list = []

    # Creating a list of all possible actions of student agent on the chessboard
    possible_actions = util.get_possible_actions(student_action_size)

    for e in range(EPISODES):
        student_agent = student_list[e % len(student_list)] #Always uses the only student at 0 if teacher_tune
        start_time = time.time()
        print_game = False
        check_mated_yet = False
        print ("episode: ", e)
        deep = Engine(depth=20) # Initialize Stockfish
        final_score = 0 # Initialize final score for the game
        done = False
        pos = game.Position(util.initial, 0, (True,True), (True,True), 0, 0)
        searcher = game.Searcher()
        moves_list = []
        round = 0
        while True:
            round += 1
            if print_game:
                game.print_pos(pos)
            state = util.toBit(pos)
            before_output_list = deep.bestmove()['info'].split(" ")
            if 'mate' in before_output_list:
                #if util.inCheck(pos, True):
                #    print("You lost, but you're getting there little one")
                #else:
                #    print("Huh.  Stalemate.  ")
                end_time = time.time()
                duration = end_time - start_time
                #print("episode: {}/{}, number of rounds: {}, score: {}, e: {:.2}, time : {}"
                      #.format(e, EPISODES, round, final_score / float(round), student_agent.epsilon, duration / 60.0))
                scores_list.append(final_score / float(round))
                #print (scores_list)
                done = True
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
                #if util.inCheck(pos, True):
                #    print("You lost, but you're getting there little one")
                #else:
                #    print("Huh.  Stalemate.  ")
                end_time = time.time()
                duration = end_time - start_time
                #print("episode: {}/{}, number of rounds: {}, score: {}, e: {:.2}, time : {}"
                #      .format(e, EPISODES, round, final_score / float(round), student_agent.epsilon, duration / 60.0))
                scores_list.append(final_score / float(round))
                #print (scores_list)
                done = True
                break
            ''' End check for check code'''

            dqn_move_index = student_agent.act(state, valid_move_indices)

            ''' TEACHER '''
            if tune_teacher:
                copy_moves_list = moves_list[:]
                teacher_state, optimal_piece_move_indices_maybe, best_move_index, had_a_nan = util.getTeacherState(dqn_move_index, valid_move_indices, possible_actions, copy_moves_list, deep)
                if print_game:
                    print('teacher state: ', teacher_state)
                if had_a_nan:
                    game.print_pos(pos)

                # get teacher action
                teacher_action_index = teacher_agent.act(teacher_state)
                if len(optimal_piece_move_indices_maybe) == 0:
                    teacher_action_index = 0
                    print('error avoided')
                if teacher_action_index == 1:
                    move_index_based_on_partial = student_agent.act(state, optimal_piece_move_indices_maybe)
                    while move_index_based_on_partial not in optimal_piece_move_indices_maybe:
                         move_index_based_on_partial = random.choice(optimal_piece_move_indices_maybe)
                    dqn_move_index = move_index_based_on_partial
                    if print_game:
                        print("Partial mint, chocolate chip mint")
                    optimal_piece_moves = []
                    for i in optimal_piece_move_indices_maybe:
                        optimal_piece_moves.append(possible_actions[i])
                    if print_game:
                        print("optimal piece moves: ", optimal_piece_moves)
                elif teacher_action_index == 2:
                    if print_game:
                        print("Full hint: ", possible_actions[best_move_index])
                    dqn_move_index = best_move_index
                else:
                    if print_game:
                        print("Not a bit of a hint (no hint)")
                    assert teacher_action_index == 0
                ''' TEACHER '''

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
                #if util.inCheck(pos, True):
                #    print("You lost, but you're getting there little one")
                #else:
                #    print("Huh.  Stalemate.  ")
                end_time = time.time()
                duration = end_time - start_time
                #print("episode: {}/{}, number of rounds: {}, score: {}, e: {:.2}, time : {}"
                #      .format(e, EPISODES, round, final_score / float(round), student_agent.epsilon, duration / 60.0))
                scores_list.append(final_score / float(round))
                #print (scores_list)
                done = True
                new_state = util.toBit(pos.getNewState(dqn_move))
                student_agent.remember(state, dqn_move_index, -5000, new_state, done)
                break
            else:
                score_after_model_move = (-1)*int(after_output['info'].split(" ")[9]) # changed from 9

            # Q-Learning
            pos.rotate()
            new_state = util.toBit(pos.getNewState(dqn_move))
            reward = score_after_model_move - score_before_model_move
            final_score += reward
            student_agent.remember(state, dqn_move_index, reward, new_state, done)
            state = new_state

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
            #         reward = 300.0 + score_student - score_optimal #Use ETA if teacher_action_index = 1
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
                print (scores_list)
                done = True
                break

            ''' End check for check code'''

            ''' if there is a problem in the future with valid moves, it might be because sunfish moves into check '''

            # Opponent takes an action
            opponent_move, score = searcher.search(pos, secs=2)
            opponent_move_stockfish = game.render(119-opponent_move[0]) + game.render(119-opponent_move[1])
            pos = pos.move(opponent_move, False)
            moves_list.append(opponent_move_stockfish)
            deep.setposition(moves_list)

            # take care of replay
            if len(student_agent.memory) > student_agent.batch_size:
                student_agent.replay(student_agent.batch_size)

            # if len(teacher_agent.memory) > batch_size:
            #     teacher_agent.replay(batch_size)

        # save teacher every name
        # if with_teacher:
        #     student_name = 'save/with_teacher_' + str(e) + '.h5'
        # else:
        # student_name = 'save/without_teacher_' + str(e) + '.h5'
        # print("saving " + student_name)
        # student_agent.save(student_name)
        if tune_teacher: #Not needed, but in for clarity
            student_list[0] = student_agent

    return sum(scores_list) / (EPISODES + 0.0)

###############################################################################
# SGD
###############################################################################

if __name__ == "__main__":
    tune_teacher = True
    teacher_agent = models.TeacherAgent()
    student_agent = models.StudentAgent()
    if tune_teacher:
        teacher_agent.load('save/teacher.h5')
    else:
        version_num = 125
        student_start_file = "save/without_teacher_" + str(version_num) + ".h5"
        student_agent.load(student_start_file)
    pos = game.Position(util.initial, 0, (True,True), (True,True), 0, 0)
    searcher = game.Searcher()
    deep = Engine(depth=20)
    def assign_params(params):
        teacher_agent.learning_rate = params[0]
        teacher_agent.batch_size = 2 ** int(params[1]) #Need to add batch size to studentAgent class
        teacher_agent.gamma = params[2]
    ranges = [0.004, 5, 0.2] #Use this so that we can choose powers of two as batch sizes
    mins = [0.001, 1, 0.8]
    maxes = [0.005, 6, 1.0]
    def init_params():
        init_parameters = []
        for i in range(3): #3 = num parameters, lr, bs, gamma
            init_parameters.append(random.random() * ranges[i] + mins[i])
        init_parameters[1] = random.randint(1, 6) # Set batch size discretely
        return init_parameters
    params = init_params()
    print("starting params: ", params)
    if tune_teacher:
        param_file_name = "teacher_params.txt"
    else:
        param_file_name = "student_params.txt"
    best_average_score = -2000
    num_tries = 0
    max_tries = 50
    target_score = 200
    while best_average_score < target_score and num_tries < max_tries:
        print("current parameters: ", params)
        num_tries += 1
        for i in range(len(params)):
            print("Exploring param ", str(i), " with value ", str(params[i]))
            increase_score = -2000
            if i == 1:
                higher_val = params[i] + 1
            else:
                higher_val = params[i] + 0.05 * ranges[i]
            if higher_val <= maxes[i]:
                print("Increasing parameter ", str(i), " from ", str(params[i]), " to ", str(higher_val))
                #print("Increasing from " + str(params[i]) + " to " + str(higher_val))

                params[i] = higher_val
                assign_params(params)
                increase_score = get_four_game_average_score(teacher_agent, student_agent, tune_teacher)
                if i == 1:
                    params[i] -= 1
                else:
                    params[i] -= 0.05 * ranges[i]
            decrease_score = -2000
            if i == 1:
                lower_val = params[i] - 1
            else:
                lower_val = params[i] - 0.05 * ranges[i]
            if lower_val >= mins[i]:
                #print("Decreasing from " + str(params[i]) + " to " + str(lower_val))
                print("Decreasing parameter ", str(i), " from ", str(params[i]), " to ", str(lower_val))
                params[i] = lower_val
                assign_params(params)
                decrease_score = get_four_game_average_score(teacher_agent, student_agent, tune_teacher)
                if i == 1:
                    params[i] += 1
                else:
                    params[i] += 0.05 * ranges[i]
            print("Old best score: ", str(best_average_score))
            print("Increase score: ", str(increase_score))
            print("Decrease score: ", str(decrease_score))
            if max(increase_score, decrease_score) > best_average_score:
                if increase_score > decrease_score:
                    best_average_score = increase_score
                    params[i] += 0.05 * ranges[i]
                else:
                    best_average_score = decrease_score
                    params[i] -= 0.05 * ranges[i]
        print("final parameters: ")
        print(params)
        with open(param_file_name, "a") as write_file:
            for i in range(len(params)):
                write_file.write(str(params[i]))
                write_file.write("/////")

# student_agent = StudentAgent()
# teacher_agent = TeacherAgent()
# student_net = KerasClassifier(build_fn=student_agent._build_model, verbose=0)
# learning_rates = [0.001, 0.002, 0.003, 0.004, 0.005]
# batch_sizes = [2, 4, 8, 16, 32]
# gammas = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95]
# hyperparameters = dict(sgd_learning_rate = learning_rates, sgd_batch_size = batch_sizes, sgd_gamma = gammas)
# grid = GridSearchCV(estimator=student_net, param_grid=hyperparameters)
# grid_result = grid.fit(features, target)
# best_params = grid_result.best_params_
