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
    episodes_per_student = 1
    EPISODES = 135
    student_action_size = 1856
    start_episode = 117
    teacher_agent = models.TeacherAgent()
    student_agent = models.StudentAgent()
    teacher_agent.load('save/teacher.h5')
    if with_teacher:
        filename = 'save/with_teacher_' + str(start_episode) + '.h5'
    else:
        filename = 'save/without_teacher_' + str(start_episode) + '.h5'
    student_agent.load(filename)
    batch_size = 8
    scores_list = []
    old_scores_list = []
    hints_list = []
    partial_hint_list = []

    # Creating a list of all possible actions of student agent on the chessboard
    possible_actions = util.get_possible_actions(student_action_size)

    for e in range(start_episode + 1, start_episode + 1 + EPISODES + 1):
        partial_diffs_for_game = []
        hints_for_game = {0:0, 1:0, 2:0}
        start_time = time.time()
        print_game = (e + 1) % 25 == 0
        check_mated_yet = False
        print ("episode: ", e)
        deep = Engine(depth=20) # Initialize Stockfish
        final_score = 0 # Initialize final score for the game
        old_final_score = 0
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
                if util.inCheck(pos, True):
                    print("You lost, but you're getting there little one")
                else:
                    print("Huh.  Stalemate.  ")
                end_time = time.time()
                duration = end_time - start_time
                print("episode: {}/{}, number of rounds: {}, score: {}, e: {:.2}, time : {}"
                      .format(e, EPISODES, round, final_score / float(round), student_agent.epsilon, duration / 60.0))
                scores_list.append(final_score / float(round))
                print (scores_list)
                old_scores_list.append(old_final_score / float(round))
                print (old_scores_list)
                hints_list.append(hints_for_game)
                print (hints_list)
                partial_hint_list.append(partial_diffs_for_game)
                print (partial_hint_list)
                done = True
                break
            else:
                score_before_model_move = int(before_output_list[9]) # changed from 9

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
                print (scores_list)
                old_scores_list.append(old_final_score / float(round))
                print (old_scores_list)
                hints_list.append(hints_for_game)
                print (hints_list)
                partial_hint_list.append(partial_diffs_for_game)
                print (partial_hint_list)
                done = True
                break
            ''' End check for check code'''

            dqn_move_index = student_agent.act(state, valid_move_indices)

            ''' TEACHER '''
            if with_teacher:
                copy_moves_list = moves_list[:]
                teacher_state, optimal_piece_move_indices_maybe, best_move_index, had_a_nan = util.getTeacherState(dqn_move_index, valid_move_indices, possible_actions, copy_moves_list, deep)
                if print_game:
                    print('teacher state: ', teacher_state)
                if had_a_nan:
                    game.print_pos(pos)

                # get teacher action
                teacher_action_index = teacher_agent.act(teacher_state)
                hints_for_game[teacher_action_index] += 1
                if len(optimal_piece_move_indices_maybe) == 0:
                    teacher_action_index = 0
                    print('error avoided')
                if teacher_action_index == 1:
                    move_index_based_on_partial = student_agent.act(state, optimal_piece_move_indices_maybe)
                    while move_index_based_on_partial not in optimal_piece_move_indices_maybe:
                         move_index_based_on_partial = random.choice(optimal_piece_move_indices_maybe)
                    dqn_move_index = move_index_based_on_partial
                    diff = util.get_move_value(best_move_index, moves_list, possible_actions, deep) - util.get_move_value(dqn_move_index, moves_list, possible_actions, deep)
                    partial_diffs_for_game.append(diff)
                    if print_game:
                        print("Partial mint, chocolate chip mint")
                    optimal_piece_moves = []
                    for i in optimal_piece_move_indices_maybe:
                        optimal_piece_moves.append(possible_actions[i])
                    if print_game:
                        print("optimal piece moves: ", optimal_piece_moves)
                elif teacher_action_index == 2:
                    partial_diffs_for_game.append(None)
                    if print_game:
                        print("Full hint: ", possible_actions[best_move_index])
                    dqn_move_index = best_move_index
                else:
                    partial_diffs_for_game.append(None)
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
                if util.inCheck(pos, True):
                    print("You lost, but you're getting there little one")
                else:
                    print("Huh.  Stalemate.  ")
                end_time = time.time()
                duration = end_time - start_time
                print("episode: {}/{}, number of rounds: {}, score: {}, e: {:.2}, time : {}"
                      .format(e, EPISODES, round, final_score / float(round), student_agent.epsilon, duration / 60.0))
                scores_list.append(final_score / float(round))
                print (scores_list)
                old_scores_list.append(old_final_score / float(round))
                print (old_scores_list)
                hints_list.append(hints_for_game)
                print (hints_list)
                partial_hint_list.append(partial_diffs_for_game)
                print (partial_diffs_for_game)
                partial_hint_list.append(partial_diffs_for_game)
                print (partial_hint_list)
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
            old_final_score += (score_after_model_move + score_before_model_move)
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
                print (scores_list)
                old_scores_list.append(old_final_score / float(round))
                print (old_scores_list)
                hints_list.append(hints_for_game)
                print (hints_list)
                partial_hint_list.append(partial_diffs_for_game)
                print (partial_hint_list)
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
        if with_teacher:
            student_name = 'save/with_teacher_' + str(e) + '.h5'
        else:
            student_name = 'save/without_teacher_' + str(e) + '.h5'
        print("saving " + student_name)
        student_agent.save(student_name)
