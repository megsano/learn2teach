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
                matched_list.append(float(matched) / float(round))
                rounds_list.append(round)
                mat_diff = util.get_material_difference(pos)
                mat_diff_list.append(mat_diff)
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
                matched_list.append(float(matched) / float(round))
                rounds_list.append(round)
                mat_diff = util.get_material_difference(pos)
                mat_diff_list.append(mat_diff)
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

            # STUDENT ACTUALLY ACTS #
            dqn_move = possible_actions[dqn_move_index]
            # flip move
            flipped_dqn_move = util.flip_move(dqn_move)
            pos = pos.move(dqn_move, True) ## used to be new_dqn_move
            # update stockfish based on DQN action
            dqn_move_stockfish = game.render(119-flipped_dqn_move[0]) + game.render(119-flipped_dqn_move[1]) ## used to be dqn_move
            moves_list.append(dqn_move_stockfish)
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
                matched_list.append(float(matched) / float(round))
                rounds_list.append(round)
                mat_diff = util.get_material_difference(pos)
                mat_diff_list.append(mat_diff)
                done = True
                break
            else:
                score_after_model_move = (-1)*int(after_output['info'].split(" ")[9]) # changed from 9

            # Q-Learning
            pos.rotate()
            reward = score_after_model_move - score_before_model_move
            final_score += reward

            if print_game:
                game.print_pos(pos.rotate())
            else:
                pos.rotate()

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
