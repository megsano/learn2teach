# -*- coding: utf-8 -*-
import random
import math
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.models import load_model
import h5py
import time
import game
from pystockfish import *

###############################################################################
# Helper functions
###############################################################################
'''
Input: move index
Output: value of the move
'''

def get_move_value(move_index, moves_list, possible_actions, deep):
    before_output = deep.bestmove() # computing score of board before student agent makes action
    before_output_list = before_output['info'].split(" ")
    score_before_that_move = (-1)*int(before_output_list[9]) # changed from 9
    move_tuple = possible_actions[move_index]
    move_for_stockfish = game.render(119-move_tuple[0]) + game.render(119-move_tuple[1])
    moves_list.append(move_for_stockfish)
    deep.setposition(moves_list) #SYNTAX!!
    after_output = deep.bestmove()
    after_output_list = after_output['info'].split(" ")
    score_after_that_move = (-1)*int(after_output_list[9]) # changed from 9
    last_thing = moves_list.pop()
    deep.setposition(moves_list) #Does set and reset deep position -- probably not buggy, but look into
    return score_after_that_move - score_before_that_move

'''
Convert "e2" to "85", that kind of thing
'''
def convert_to_nums(algebra_str):
    #print("algebra str: ", algebra_str)
    letter = algebra_str[0]
    number = algebra_str[1]
    letter_as_num = ord(letter) - ord("a") + 1
    try:
        new_number = 10 - int(number)
        return 10 * new_number + letter_as_num
    except:
        return None
    #print("converted: ", 10 * new_number + letter_as_num)

'''
Get teacher state
'''
def getTeacherState(suggested_move_index, valid_move_indices, possible_actions, moves_list, deep): #UPDATE PARAMS IN MAIN!!!
    had_a_nan_in_teacher_state = False
    state = []
    # state 1: difference between suggested move value and optimal move value
    output = deep.bestmove() ## something else
    #print("optimal move now: ", output['move'])
    best_move = (convert_to_nums(output['move'][0:2]),convert_to_nums(output['move'][2:]))
    if best_move[0] == None or best_move[1] == None:
        best_move_index = random.choice(valid_move_indices)
        best_move = possible_actions[best_move_index]
    else:
        best_move_index = possible_actions.index(best_move)
    suggested_move_value = get_move_value(suggested_move_index, moves_list, possible_actions, deep)
    optimal_move_value = get_move_value(best_move_index, moves_list, possible_actions, deep)
    diff = suggested_move_value - optimal_move_value
    state.append(diff)
    # state 2: optimal move
    valid_moves_values = []# get value of move for move in possibly_valid_move_indices
    for valid_move_index in valid_move_indices:
        valid_moves_values.append(get_move_value(valid_move_index, moves_list, possible_actions, deep))
    state.append(np.std(np.array(valid_moves_values)))
    # state 3: boolean comparing suggested and optimal
    state.append(best_move_index == suggested_move_index)
    # state 4: std of optimal piece moves
    optimal_piece_move_indices = []
    #piece_to_move_loc = best_move[0:2]# in the form of
    reformatted_loc = best_move[0]# e2 --> 85
    #print ("whether best move is in valid move indices", best_move_index in valid_move_indices)
    for move_index in valid_move_indices:
        if possible_actions[move_index][0] == reformatted_loc:
            optimal_piece_move_indices.append(move_index)
    if len(optimal_piece_move_indices) == 0:
        print ("length of optimal piece move indices: ", 0)
        print (valid_move_indices)
    #print (optimal_piece_move_indices)
    optimal_piece_move_values = []
    for optimal_piece_move_index in optimal_piece_move_indices:
        optimal_piece_move_values.append(get_move_value(optimal_piece_move_index, moves_list, possible_actions, deep))
    if len(optimal_piece_move_values) == 0:
        had_a_nan_in_teacher_state = True
    state.append(np.std(np.array(optimal_piece_move_values)))
    # state 5: number of moves since partial hint or no hint
    #state.append(teacher_agent.moves_since_hint)
    if np.isnan(state[3]):
        had_a_nan_in_teacher_state = True
    state = np.reshape(state, (1,4))
    #print('teacher state: ', state)
    return state, optimal_piece_move_indices, best_move_index, had_a_nan_in_teacher_state


'''
Converts the 64-character representation of the board state used in
game.py to a 64-bit representation.
'''

def toBit(pos):
    board = pos.board
    wc = pos.wc
    bc = pos.bc
    ep = pos.ep
    kp = pos.kp
    state = [0] * 18
    indices = ['P', 'R', 'N', 'B', 'Q', 'K', 'p', 'r', 'n', 'b', 'q', 'k']
    rows = "".join(board.split('\n')[:-1]) ## might be a bug
    for i in range(len(rows)):
        piece = rows[i]
        for j, letter in enumerate(indices):
            if piece == letter:
                state[j] += 2 ** i
    wc_1, wc_2 = wc
    bc_1, bc_2 = bc
    state[12] = int(wc_1)
    state[13] = int(wc_2)
    state[14] = int(bc_1)
    state[15] = int(bc_2)
    state[16] = int(ep)
    state[17] = int(kp)
    state = np.reshape(state, (1,18))
    return state

'''
Retrieves the king's position at any given board state.
'''

def getKingPos(boardString, piece_letter):
    #print(piece_letter)
    # boardString_list = []
    # for row in position.board.split("\n"):
    #     if row != "" and not row.isspace():
    #         boardString_list.append(row)
    # boardString = "".join(boardString_list)#position.board.split("\n")[3:11])
    # #print("length of board string before reducing empty spaces: ", str(len(boardString)))
    # boardString = boardString.replace(" ", "")
    #print(splitted)
    if len(boardString) != 64:
        print("boardString length = " + str(len(boardString)))
    #print ("boardString: ", boardString)
    splitted = []
    for i in range(8):
        splitted.append(boardString[8 * i : 8 * i + 8])
    for l, line in enumerate(splitted): # "".join(board.split('\n')[:-1])
        #print(line)
        for col in range(8):
            #print (line[col])
            if line[col] == piece_letter:
                return (l, col)
    return ("nope", "nope")

'''
Checks if the input player is in check.
'''
def inCheck(position, checkCurrentPlayer):
    pieceLetters = ["P", "N", "B", "R", "Q", "K"]
    king = 'k'
    if checkCurrentPlayer:
        pieceLetters = ["p", "n", "b", "r", "q", "k"] ## case is same for opponent and king
        king = 'K'
    #print(king)
    # pieceLetters = ["p", "n", "b", "r", "q", "k"]
    # if not checkCurrentPlayer:
    #     pieceLetters = ["P", "N", "B", "R", "Q", "K"] ## case is same for opponent and king
     ## changed from pieceLetters[-1] # it's when it's lowercase that getKingPos error happens
    #print(position.board.split("\n"))
    boardString_list = []
    for row in position.board.split("\n"):
        if row != "" and not row.isspace():
            boardString_list.append(row)
    boardString = "".join(boardString_list)#position.board.split("\n")[3:11])
    #print("length of board string before reducing empty spaces: ", str(len(boardString)))
    boardString = boardString.replace(" ", "")
    #print("length of board string: ", str(len(boardString)))
    kRow, kCol = getKingPos(boardString, king)
    if kRow == "nope": ## somewhat big change
        return True
    for row in range(8):
        for col in range(8):
            pieceIndex = 8 * row + col
            piece = boardString[pieceIndex]
            if piece == pieceLetters[0]:
                if row + 1 == kRow and abs(col - kCol) == 1:
                    # print(pieceLetters[0])
                    return True
            if piece == pieceLetters[1]:
                if abs(row - kRow) == 2 and abs(col - kCol) == 1:
                    # print(pieceLetters[1])
                    return True
                if abs(row - kRow) == 1 and abs(col - kCol) == 2:
                    # print(pieceLetters[1])
                    return True
            if piece == pieceLetters[2] or piece == pieceLetters[4]:
                if abs(row - kRow) == abs(col - kCol):
                    canCheck = True
                    if (row - kRow) * (col - kCol) > 0:
                        start = 8 * min(row, kRow) + min(col, kCol)
                        for diagonAlley in range(abs(row - kRow) - 1):
                            # if boardString[start + 9 * diagonAlley] != ".":
                            #     canCheck = False
                            if boardString[start + 9 * (diagonAlley + 1)] != ".":
                                canCheck = False
                        if canCheck == True:
                            # print(piece)
                            return True
                    else:
                        start = 8 * min(row, kRow) + max(col, kCol)
                        for diagonAlley in range(abs(row - kRow) - 1):
                            if boardString[start + 7 * (diagonAlley + 1)] != ".":
                                canCheck = False
                        if canCheck == True:
                            # print(piece)
                            return True
            if piece == pieceLetters[3] or piece == pieceLetters[4]:
                if row == kRow:
                    canCheck = True
                    for inBetween in range(min(col, kCol) + 1, max(col, kCol)):
                        if boardString[8 * row + inBetween] != ".":
                            canCheck = False
                    if canCheck == True:
                        # print(piece)
                        return True
                if col == kCol:
                    canCheck = True
                    for inBetween in range(min(row, kRow) + 1, max(row, kRow)):
                        if boardString[8 * inBetween + col] != ".":
                            canCheck = False
                    if canCheck == True:
                        # print(piece)
                        return True
            if piece == pieceLetters[5]:
                if abs(row - kRow) <= 1 and abs(col - kCol) <= 1:
                    # print(pieceLetters[5])
                    return True
    return False

###########################
def get_possible_actions(student_action_size):
    possible_actions = []
    for x_prev in range(2,10):
        for y_prev in range(1,9):
            for x_next in range(2,10):
                for y_next in range(1,9):
                    if x_next == x_prev or y_next == y_prev or abs(x_prev - x_next) == abs(y_prev - y_next):
                        possible_actions.append((int(str(x_prev) + str(y_prev)), int(str(x_next) + str(y_next))))
                    elif abs(x_prev - x_next) <= 1 and abs(y_prev - y_next) <= 1:
                        possible_actions.append((int(str(x_prev) + str(y_prev)), int(str(x_next) + str(y_next))))
                    elif abs(x_prev - x_next) == 1 and abs(y_prev - y_next) == 2:
                        possible_actions.append((int(str(x_prev) + str(y_prev)), int(str(x_next) + str(y_next))))
                    elif abs(x_prev - x_next) == 2 and abs(y_prev - y_next) == 1:
                        possible_actions.append((int(str(x_prev) + str(y_prev)), int(str(x_next) + str(y_next))))
    assert len(possible_actions) == student_action_size
    return possible_actions



def flip_move(dqn_move):
    firstPos = dqn_move[0]
    firstPosRow = int(str(firstPos)[0])
    firstPosCol = int(str(firstPos)[1])
    secondPos = dqn_move[1]
    secondPosRow = int(str(secondPos)[0])
    secondPosCol = int(str(secondPos)[1])
    newFirstPosRow = 11 - firstPosRow
    newFirstPosCol = 9 - firstPosCol
    newSecondPosRow = 11 - secondPosRow
    newSecondPosCol = 9 -secondPosCol
    new_dqn_move = (int(str(newFirstPosRow) + str(newFirstPosCol)), int(str(newSecondPosRow)+str(newSecondPosCol)))
    return new_dqn_move

'''Had a nice shirt on Wednesday:'''
def percy_liang(move_list, old_pos):
    new_pos = game.Position(initial, 0, (True,True), (True,True), 0, 0) # game.py stuff
    for i in range(len(move_list)):
        player_bool = not i % 2 == 0 # used to be not
        move_string = move_list[i]
        move_nums = (convert_to_nums(move_string[0:2]),convert_to_nums(move_string[2:]))
        new_pos = new_pos.move(move_nums, player_bool)
    # print("move list passed into percy: ")
    # print(move_list)
    # print("new: ", new_pos)
    # print("old", old_pos)
    return new_pos.board == old_pos.board




###############################################################################
# Variables and constants for sunfish
###############################################################################

piece = { 'P': 100, 'N': 280, 'B': 320, 'R': 479, 'Q': 929, 'K': 60000 }
pst = {
    'P': (   0,   0,   0,   0,   0,   0,   0,   0,
            78,  83,  86,  73, 102,  82,  85,  90,
             7,  29,  21,  44,  40,  31,  44,   7,
           -17,  16,  -2,  15,  14,   0,  15, -13,
           -26,   3,  10,   9,   6,   1,   0, -23,
           -22,   9,   5, -11, -10,  -2,   3, -19,
           -31,   8,  -7, -37, -36, -14,   3, -31,
             0,   0,   0,   0,   0,   0,   0,   0),
    'N': ( -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6, 100, -36,   4,  62,  -4, -14,
            10,  67,   1,  74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,   5,  31,  21,  22,  35,   2,   0,
           -18,  10,  13,  22,  18,  15,  11, -14,
           -23, -15,   2,   0,   2,   0, -23, -20,
           -74, -23, -26, -24, -19, -35, -22, -69),
    'B': ( -59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10),
    'R': (  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -24, -18,   5,  -2, -18, -31, -32),
    'Q': (   6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -10,  20,  76,  57,  24,
            -2,  43,  32,  60,  72,  63,  43,   2,
             1, -16,  22,  17,  25,  20, -13,  -6,
           -14, -15,  -2,  -5,  -1, -10, -20, -22,
           -30,  -6, -13, -11, -16, -11, -16, -27,
           -36, -18,   0, -19, -15, -15, -21, -38,
           -39, -30, -31, -13, -31, -36, -34, -42),
    'K': (   4,  54,  47, -99, -99,  60,  83, -62,
           -32,  10,  55,  56,  56,  55,  10,   3,
           -62,  12, -57,  44, -67,  28,  37, -31,
           -55,  50,  11,  -4, -19,  13,   0, -49,
           -55, -43, -52, -28, -51, -47,  -8, -50,
           -47, -42, -43, -79, -64, -32, -29, -32,
            -4,   3, -14, -50, -57, -18,  13,   4,
            17,  30,  -3, -14,   6,  -1,  40,  18),
}
# Pad tables and join piece and pst dictionaries
for k, table in pst.items():
    padrow = lambda row: (0,) + tuple(x+piece[k] for x in row) + (0,)
    pst[k] = sum((padrow(table[i*8:i*8+8]) for i in range(8)), ())
    pst[k] = (0,)*20 + pst[k] + (0,)*20

# Our board is represented as a 120 character string. The padding allows for
# fast detection of moves that don't stay within the board.
A1, H1, A8, H8 = 91, 98, 21, 28
initial = (
    '         \n'  #   0 -  9
    '         \n'  #  10 - 19
    ' rnbqkbnr\n'  #  20 - 29
    ' pppppppp\n'  #  30 - 39
    ' ........\n'  #  40 - 49
    ' ........\n'  #  50 - 59
    ' ........\n'  #  60 - 69
    ' ........\n'  #  70 - 79
    ' PPPPPPPP\n'  #  80 - 89
    ' RNBQKBNR\n'  #  90 - 99
    '         \n'  # 100 -109
    '         \n'  # 110 -119
)

# Lists of possible moves for each piece type.
N, E, S, W = -10, 1, 10, -1
directions = {
    'P': (N, N+N, N+W, N+E),
    'N': (N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W),
    'B': (N+E, S+E, S+W, N+W),
    'R': (N, E, S, W),
    'Q': (N, E, S, W, N+E, S+E, S+W, N+W),
    'K': (N, E, S, W, N+E, S+E, S+W, N+W)
}

# Mate value must be greater than 8*queen + 2*(rook+knight+bishop)
# King value is set to twice this value such that if the opponent is
# 8 queens up, but we got the king, we still exceed MATE_VALUE.
# When a MATE is detected, we'll set the score to MATE_UPPER - plies to get there
# E.g. Mate in 3 will be MATE_UPPER - 6
MATE_LOWER = piece['K'] - 10*piece['Q']
MATE_UPPER = piece['K'] + 10*piece['Q']

# The table size is the maximum number of elements in the transposition table.
TABLE_SIZE = 1e8

# Constants for tuning search
QS_LIMIT = 150
EVAL_ROUGHNESS = 20
