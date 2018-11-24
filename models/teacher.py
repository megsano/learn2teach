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

# from keras import models
# from keras import layers
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.datasets import make_classification
import game
from pystockfish import *
random.seed(3)
np.random.seed(3)


########### TEACHING AGENT #############################

class TeacherAgent:
    def __init__(self, state_size=4, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.moves_since_hint = 0
        self.not_yet_rewarded = []

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if self.moves_since_hint == 2: #Meaning the teacher waits at most 3 moves for a reward, could be tweaked
            self.moves_since_hint = 0
            return 2 #This enforces that we don't go too long without giving hints (could change to full OR partial hint l8r)
        if np.random.rand() <= self.epsilon:
            random_index = random.randrange(self.action_size)
            if random_index == 0:
                self.moves_since_hint += 1
            else:
                self.moves_since_hint = 0
            return random_index
        act_values = self.model.predict(state)
        nonrandom_index = np.argmax(act_values[0])  # returns action
        if nonrandom_index == 0:
            self.moves_since_hint += 1
        else:
            self.moves_since_hint = 0
        return nonrandom_index

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            # if not done: #USED TO BE COMMENTED IN
            #print ("teacher agent state (should be an array with shape (4, )): ", next_state)
            # target = (reward + self.gamma *
            #           np.amax(self.model.predict(next_state)[0]))
            whole_list = self.model.predict(next_state)
            amax_result = np.amax(whole_list[0])
            target = reward + self.gamma * amax_result
            # tabbed over part above

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


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

###############################################################################
# DQN Model
###############################################################################

class StudentAgent:
    def __init__(self, state_size=18, action_size=1856):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.5 # exploration rate
        self.epsilon_min = 0.36
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.batch_size = 32

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_move_indices):
        #print (len(valid_move_indices))
        if np.random.rand() <= self.epsilon:
            random_action_index = random.randrange(self.action_size)
            if random_action_index in valid_move_indices:
                return random_action_index
            # while random_action_index not in valid_move_indices:
            #     random_action_index = random.randrange(self.action_size)
            # return random_action_index
        act_values = self.model.predict(state)
        new_act_values = []
        for i,val in enumerate(act_values[0]):
            if i in valid_move_indices:
                new_act_values.append(val)
            else:
                new_act_values.append(0.0)
        startIndex = random.randint(0, len(new_act_values) - 1)
        newVals = []
        for i in range(len(new_act_values) - startIndex):
            newVals.append(new_act_values[startIndex + i])
        for j in range(startIndex):
            newVals.append(new_act_values[j])
        rotated_index = np.argmax(newVals)
        return (rotated_index + startIndex) % len(new_act_values)
        # return np.argmax(new_act_values)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            # if not done: USED TO BE NOT COMMENTED OUT
                #print ("student agent state length (should be 18): ", len(next_state))
            target = (reward + self.gamma *
                      np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

###############################################################################
# Helper functions
###############################################################################
'''
Input: move index
Output: value of the move
'''

def get_move_value(move_index, moves_list, possible_actions):
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
    new_number = 10 - int(number)
    #print("converted: ", 10 * new_number + letter_as_num)
    return 10 * new_number + letter_as_num
'''
Get teacher state
'''
def getTeacherState(suggested_move_index, valid_move_indices, possible_actions, moves_list): #UPDATE PARAMS IN MAIN!!!
    had_a_nan_in_teacher_state = False
    state = []
    # state 1: difference between suggested move value and optimal move value
    output = deep.bestmove() ## something else
    print("optimal move now: ", output['move'])
    best_move = (convert_to_nums(output['move'][0:2]),convert_to_nums(output['move'][2:]))
    best_move_index = possible_actions.index(best_move)
    suggested_move_value = get_move_value(suggested_move_index, moves_list, possible_actions)
    optimal_move_value = get_move_value(best_move_index, moves_list, possible_actions)
    diff = suggested_move_value - optimal_move_value
    state.append(diff)
    # state 2: optimal move
    valid_moves_values = []# get value of move for move in possibly_valid_move_indices
    for valid_move_index in valid_move_indices:
        valid_moves_values.append(get_move_value(valid_move_index, moves_list, possible_actions))
    state.append(np.std(np.array(valid_moves_values)))
    # state 3: boolean comparing suggested and optimal
    state.append(best_move_index == suggested_move_index)
    # state 4: std of optimal piece moves
    optimal_piece_move_indices = []
    #piece_to_move_loc = best_move[0:2]# in the form of
    reformatted_loc = best_move[0]# e2 --> 85
    print ("whether best move is in valid move indices", best_move_index in valid_move_indices)
    for move_index in valid_move_indices:
        if possible_actions[move_index][0] == reformatted_loc:
            optimal_piece_move_indices.append(move_index)
    if len(optimal_piece_move_indices) == 0:
        print ("length of optimal piece move indices: ", 0)
        print (valid_move_indices)
    print (optimal_piece_move_indices)
    optimal_piece_move_values = []
    for optimal_piece_move_index in optimal_piece_move_indices:
        optimal_piece_move_values.append(get_move_value(optimal_piece_move_index, moves_list, possible_actions))
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


###############################################################################
# Training
###############################################################################
#TODOO:
#MAKE init_params completed
#MAKE ranges completed
#Make maxes completed
#MAKE mins completed
#MAKE assign_params pretty much completed (just need to make batch size a class var)
#MAKE get_four_game_average_score Not done: involves calling the original main function for 4 iterations
#parameter order: learning_rate, batch_size, gamma
if __name__ == "__main__":
    student_agent = StudentAgent()
    def assign_params(params):
        self.learning_rate = params[0]
        self.batch_size = params[1] #Need to add batch size to studentAgent class
        self.gamma = params[2]


    i_love_go_main_muse = True
    ranges = [0.004, 5, 0.2] #Use this so that we can choose powers of two as batch sizes
    mins = [0.001, 1, 0.8]
    maxes = [0.005, 6, 1.0]
    def init_params():
        init_parameters = []
        for i in range(3): #3 = num parameters, lr, bs, gamma
            init_parameters.append(random.random() * ranges[i] + mins[i])
        init_parameters[1] = random.randint(1, 6) # Set batch size discretely
    params = init_params()
    best_average_score = -2000
    num_tries = 0
    while best_average_score < -200 and num_tries < 50:
        num_tries += 1
        for i in range(len(params)):
            increase_score = -2000
            if i == 1:
                higher_val = params[i] + 1
            else:
                higher_val = params[i] + 0.02 * ranges[i]
            if higher_val <= maxes[i]:
                params[i] = higher_val
                assign_params(params)
                increase_score = get_four_game_average_score()
                params[i] -= 0.02 * ranges[i]
            decrease_score = -2000
            if i == 1:
                lower_val = params[i] - 1
            else:
                lower_val = params[i] - 0.02 * ranges[i]
            if lower_val >= mins[i]:
                params[i] = lower_val
                assign_params(params)
                decrease_score = get_four_game_average_score()
                params[i] += 0.02 * ranges[i]
            if max(increase_score, decrease_score) > best_average_score:
                if increase_score > decrease_score:
                    params[i] += 0.02 * ranges[i]
                else:
                    params[i] -= 0.02 * ranges[i]


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



 # if __name__ == "__main__":
    # Constants for training
    EPISODES = 100
    student_action_size = 1856

    # Initialize agents
    student_agent = StudentAgent()
    teacher_agent = TeacherAgent()
    batch_size = 32 # changed from 32

    # Creating a list of all possible actions of student agent on the chessboard
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


    # TEACHER ACTION LIST
    #['no_hint', 'partial_hint', 'full_hint']

    scores_list = []

    # Training EPISODES times
    for e in range(EPISODES):
        check_mated_yet = False
        print ("episode: ", e)
        deep = Engine(depth=20) # Initialize Stockfish
        final_score = 0 # Initialize final score for the game
        done = False
        pos = game.Position(initial, 0, (True,True), (True,True), 0, 0)
        searcher = game.Searcher()
        moves_list = []
        round = 0
        while True:
            round += 1
            game.print_pos(pos)
            state = toBit(pos)
            before_output = deep.bestmove() # computing score of board before student agent makes action
            before_output_list = before_output['info'].split(" ")
            if 'mate' in before_output_list:
                mate_index = before_output_list.index('mate')
                mate_score = before_output_list[mate_index + 1]
                # if not check_mated_yet:
                #     score_before_model_move = -5000 #/ float(math.sqrt(float(mate_score)))
                #     check_mated_yet = True
                # else:
                score_before_model_move = 0
                print("mate is in before_output_list so score is ", str(score_before_model_move))
            else:
                score_before_model_move = (-1)*int(before_output['info'].split(" ")[9]) # changed from 9

            # get possible valid moves of student
            possibly_valid_moves = [m for m in pos.gen_moves(False)]
            possibly_valid_move_indices = [possible_actions.index(gm) for gm in possibly_valid_moves]
            meg_is_so_so_so_so_special = True
            king_is_in_check = inCheck(pos, True) #ADDED
            if king_is_in_check:
                print("********CHECK**********")
            '''Begin check for check code'''
            valid_move_indices = []
            for index in possibly_valid_move_indices:
                newPos = pos.getNewState(possible_actions[index])
                if not inCheck(newPos, True):
                    valid_move_indices.append(index)
            if len(valid_move_indices) == 0:
                if inCheck(pos, True):
                    print("You lost, but you're getting there little one")
                    #game.print_pos(pos) #CHANGEDD think this could be a good way to tell whether game goes exactly the same way every time
                else:
                    print("Huh.  Stalemate.  ")
                print("episode: {}/{}, number of rounds: {}, score: {}, e: {:.2}"
                      .format(e, EPISODES, round, final_score / float(round), student_agent.epsilon))
                scores_list.append(final_score / float(round))
                print (scores_list)
                done = True
                break
            # else:
            #     valid_move_indices = possibly_valid_move_indices

            ''' End check for check code'''

            # DQN chooses an action
            #valid_moves = [m for m in pos.gen_moves()] #### (85,65), (87, 97)
            #valid_move_indices = [possible_actions.index(gm) for gm in valid_moves]
            dqn_move_index = student_agent.act(state, valid_move_indices)

            # a check that we should be removing later
            while dqn_move_index not in valid_move_indices:
                dqn_move_index = random.choice(valid_move_indices)

            if dqn_move_index not in valid_move_indices:
                print("made invalid move")
                break

            #suggested_dqn_move_stockfish = game.render(119-possible_actions[dqn_move_index][0]) + game.render(119-possible_actions[dqn_move_index][1]) ## used to be dqn_move
            #moves_list.append(suggested_dqn_move_stockfish)
            #print("dqn move stockfish: ", str(dqn_move_stockfish))
            #print(moves_list)

            ''' TEACHER '''
            # get teacher state given the student's suggested move index and the state of the game
            copy_moves_list = moves_list[:]
            teacher_state, optimal_piece_move_indices_maybe, best_move_index, had_a_nan = getTeacherState(dqn_move_index, valid_move_indices, possible_actions, copy_moves_list)
            print('teacher state: ', teacher_state)
            # if had_a_nan:
            #     print("Should print board")
            #     game.print_pos(pos)
            #moves_list.pop() # remove suggested move from moves list

            # get teacher action
            teacher_action_index = teacher_agent.act(teacher_state) ## add this to teacher agent class
            if teacher_action_index == 1:
                move_index_based_on_partial = student_agent.act(state, optimal_piece_move_indices_maybe)
                while move_index_based_on_partial not in optimal_piece_move_indices_maybe:
                     move_index_based_on_partial = random.choice(optimal_piece_move_indices_maybe)
                dqn_move_index = move_index_based_on_partial
                print("Partial mint, chocolate chip mint")
                optimal_piece_moves = []
                for i in optimal_piece_move_indices_maybe:
                    optimal_piece_moves.append(possible_actions[i])
                print("optimal piece moves: ", optimal_piece_moves)
                # partial hint was given, check which hint that was
                #If there's a BUGG: did we accidentally over-rotate board here?
            elif teacher_action_index == 2:
                print("Hole hint")
                print("Full hint: ", possible_actions[best_move_index])
                dqn_move_index = best_move_index
                # full hint was given, proceed with that move
            else:
                print("Not a bit of a hint (no hint)")
                assert teacher_action_index == 0
                # no hint was given, proceed with student's suggested move
            ''' TEACHER '''

            # STUDENT ACTUALLY ACTS #

            dqn_move = possible_actions[dqn_move_index]
            # flip move
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
            pos = pos.move(dqn_move, True) ## used to be new_dqn_move
            # update stockfish based on DQN action
            dqn_move_stockfish = game.render(119-new_dqn_move[0]) + game.render(119-new_dqn_move[1]) ## used to be dqn_move
            if king_is_in_check:
                print("Chose to move " + dqn_move_stockfish + "to escape")
            moves_list.append(dqn_move_stockfish)
            #print("dqn move stockfish: ", str(dqn_move_stockfish))
            #print(moves_list)
            deep.setposition(moves_list)


            # compute score of board after student agent makes action
            after_output = deep.bestmove()
            #print ("after output info: " + after_output['info'])
            after_output_list = after_output['info'].split(" ")
            if 'mate' in after_output_list:
                mate_index = after_output_list.index('mate')
                mate_score = after_output_list[mate_index + 1]
                if not check_mated_yet:
                    score_after_model_move = -5000 #/ float(math.sqrt(float(mate_score)))
                    check_mated_yet = True
                else:
                    score_after_model_move = 0
                print("mate is in after_output_list so score is ", str(score_after_model_move))
            else:
                score_after_model_move = (-1)*int(after_output['info'].split(" ")[9]) # changed from 9

            # Q-Learning
            pos.rotate()
            new_state = toBit(pos.getNewState(dqn_move))
            print ("reward: ", score_after_model_move, " - ", score_before_model_move)
            reward = score_after_model_move - score_before_model_move #abs(score_after_model_move - score_before_model_move)*(score_after_model_move - score_before_model_move)
            #print(reward)
            final_score += reward
            student_agent.remember(state, dqn_move_index, reward, new_state, done)
            state = new_state

            game.print_pos(pos.rotate())


            ''' Teacher Q-learning '''
            if teacher_action_index != 2:
                score_student = get_move_value(dqn_move_index, moves_list, possible_actions)
                optimal_move_index = possible_actions.index((convert_to_nums(after_output['move'][0:2]),convert_to_nums(after_output['move'][2:])))
                score_optimal = get_move_value(optimal_move_index, moves_list, possible_actions)
                reward = 300.0 + score_student - score_optimal #Use ETA if teacher_action_index = 1
                if len(teacher_agent.not_yet_rewarded) > 0:
                    most_recent = teacher_agent.not_yet_rewarded[-1]
                    if len(most_recent) == 3:
                        teacher_agent.remember(most_recent[0], most_recent[1], reward, teacher_state, most_recent[2])
                    else:
                        assert len(most_recent) == 4
                        teacher_agent.remember(most_recent[0], most_recent[1], most_recent[3], teacher_state, most_recent[2])
                    #Puts new state as last entry in the not yet remembered iteration list
                for not_yet_remembered_iteration in teacher_agent.not_yet_rewarded:
                    teacher_agent.remember(not_yet_remembered_iteration[0], not_yet_remembered_iteration[1], reward, teacher_state, not_yet_remembered_iteration[2])
                # teacher_agent.remember(teacher_state, teacher_action_index, reward, new_teacher_state, done)
                teacher_agent.not_yet_remembered = [[teacher_state, teacher_action_index, done, reward]]
            else:
                not_yet_remembered_list = [teacher_state, teacher_action_index, done]
                teacher_agent.not_yet_rewarded.append(not_yet_remembered_list)






            # # if checkmate happened, break out of the loop and print stats
            # if done:
            #     print("episode: {}/{}, score: {}, e: {:.2}"
            #           .format(e, EPISODES, round, student_agent.epsilon))
            #     break

            '''Begin check for check code''' # im tired!

            possibly_valid_moves = [m for m in pos.gen_moves(True)]
            #print(possibly_valid_moves)
            possibly_valid_move_indices = [possible_actions.index(gm) for gm in possibly_valid_moves]
            valid_move_indices = []
            #print("possibly valid move indices: ", str(len(possibly_valid_move_indices)))
            for index in possibly_valid_move_indices:
                newPos = pos.getNewState(possible_actions[index])
                #print(newPos.board)
                if not inCheck(newPos, True):
                    valid_move_indices.append(index)
            if len(valid_move_indices) == 0:
                if inCheck(pos, True):
                    print("Hahaha! We won.")
                else:
                    print("Hahaha! Stalemate. ")
                print("episode: {}/{}, number of rounds: {}, score: {}, e: {:.2}"
                      .format(e, EPISODES, round, final_score / float(round), student_agent.epsilon))
                scores_list.append(final_score / float(round))
                print (scores_list)
                done = True
                # if e % 10 == 0:
                #     score_list.append(final_score)
                #     current_mean = sum(scores_list) / 10.0
                #     if current_mean < previous_mean:
                #         raise Exception
                break
            # else:
            #     valid_move_indices = possibly_valid_move_indices

            ''' End check for check code'''

            ''' if there is a problem in the future with valid moves, it might be because sunfish moves into check '''

            #game.print_pos(pos)

            # Opponent takes an action
            opponent_move, score = searcher.search(pos, secs=2)
            opponent_move_stockfish = game.render(119-opponent_move[0]) + game.render(119-opponent_move[1])
            pos = pos.move(opponent_move, False)
            # update stockfish based on opponent action
            moves_list.append(opponent_move_stockfish)
            #print("opponent move stockfish: ", str(opponent_move_stockfish))
            #print(moves_list)
            deep.setposition(moves_list)


            # # check for checkmate
            # if score == MATE_UPPER:
            #
            #     print("episode: {}/{}, score: {}, e: {:.2}"
            #           .format(e, EPISODES, round, agent.epsilon))
            #     break

            # take care of replay
            if len(student_agent.memory) > batch_size:
                student_agent.replay(batch_size)

            if len(teacher_agent.memory) > batch_size:
                teacher_agent.replay(batch_size)

            # # if checkmate happened, break out of the loop and print stats
            # if done:
            #     print("episode: {}/{}, score: {}, e: {:.2}"
            #           .format(e, EPISODES, round, agent.epsilon))
            #     break
    print(scores_list)
    plt.plot(scores_list)
    plt.ylabel('average score value')
    plt.xlabel('games')
    plt.show()
    student_agent.save('save/agent.h5')
        #     ########## OLD CODE ###########
        # # if e % 10 == 0:
        # #     agent.save("/save/cartpole-dqn.h5")







######## ^ working code 



























# # -*- coding: utf-8 -*-
# import random
# import gym
# import numpy as np
# from collections import deque
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam
#
# EPISODES = 1000
#
# class DQNAgent:
#     def __init__(self, state_size, action_size):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.memory = deque(maxlen=2000)
#         self.gamma = 0.95    # discount rate
#         self.epsilon = 1.0  # exploration rate
#         self.epsilon_min = 0.01
#         self.epsilon_decay = 0.995
#         self.learning_rate = 0.001
#         self.model = self._build_model()
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
#         if np.random.rand() <= self.epsilon:
#             return random.randrange(self.action_size)
#         act_values = self.model.predict(state)
#         return np.argmax(act_values[0])  # returns action
#
#     def replay(self, batch_size):
#         minibatch = random.sample(self.memory, batch_size)
#         for state, action, reward, next_state, done in minibatch:
#             target = reward
#             if not done:
#                 target = (reward + self.gamma *
#                           np.amax(self.model.predict(next_state)[0]))
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
# if __name__ == "__main__":
#     #env = gym.make('CartPole-v1')
#     state_size = 5 # standard deviation of the moves with the optimal piece
#     action_size = 3
#     agent = DQNAgent(state_size, action_size)
#     # agent.load("./save/cartpole-dqn.h5")
#     done = False
#     batch_size = 32
#
#     for e in range(EPISODES):
#         #state = env.reset()
#         #state = np.reshape(state, [1, state_size])
#         for time in range(500):
#             # env.render()
#             action = agent.act(state)
#             next_state, reward, done, _ = env.step(action)
#             reward = reward if not done else -10
#             next_state = np.reshape(next_state, [1, state_size])
#             agent.remember(state, action, reward, next_state, done)
#             state = next_state
#             if done:
#                 print("episode: {}/{}, score: {}, e: {:.2}"
#                       .format(e, EPISODES, time, agent.epsilon))
#                 break
#             if len(agent.memory) > batch_size:
#                 agent.replay(batch_size)
#         # if e % 10 == 0:
#         #     agent.save("./save/cartpole-dqn.h5")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # # -*- coding: utf-8 -*-
# # """
# # Teacher model : a DQN model
# #
# # State: board state, which move student thinks is optimal, optimal action from stockfish
# # Actions: revealing hints (no_hint, piece_hint, move_hint, etc.)
# # Rewards: ???
# #
# # """
# #
# # # import gym
# # import math
# # import random
# # import numpy as np
# # import matplotlib
# # import matplotlib.pyplot as plt
# # from collections import namedtuple
# # from itertools import count
# # from PIL import Image
# #
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # import torch.nn.functional as F
# # import torchvision.transforms as T
# #
# #
# # # env = gym.make('CartPole-v0').unwrapped
# #
# # # set up matplotlib
# # is_ipython = 'inline' in matplotlib.get_backend()
# # if is_ipython:
# #     from IPython import display
# #
# # plt.ion()
# #
# # # if gpu is to be used
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #
# #
# # ######################################################################
# # # Replay Memory
# # # -------------
# # #
# # # We'll be using experience replay memory for training our DQN. It stores
# # # the transitions that the agent observes, allowing us to reuse this data
# # # later. By sampling from it randomly, the transitions that build up a
# # # batch are decorrelated. It has been shown that this greatly stabilizes
# # # and improves the DQN training procedure.
# # #
# # # For this, we're going to need two classses:
# # #
# # # -  ``Transition`` - a named tuple representing a single transition in
# # #    our environment
# # # -  ``ReplayMemory`` - a cyclic buffer of bounded size that holds the
# # #    transitions observed recently. It also implements a ``.sample()``
# # #    method for selecting a random batch of transitions for training.
# # #
# #
# # Transition = namedtuple('Transition',
# #                         ('state', 'action', 'next_state', 'reward'))
# #
# #
# # class ReplayMemory(object):
# #
# #     def __init__(self, capacity):
# #         self.capacity = capacity
# #         self.memory = []
# #         self.position = 0
# #
# #     def push(self, *args):
# #         """Saves a transition."""
# #         if len(self.memory) < self.capacity:
# #             self.memory.append(None)
# #         self.memory[self.position] = Transition(*args)
# #         self.position = (self.position + 1) % self.capacity
# #
# #     def sample(self, batch_size):
# #         return random.sample(self.memory, batch_size)
# #
# #     def __len__(self):
# #         return len(self.memory)
# #
# #
# # ######################################################################
# # # Now, let's define our model. But first, let quickly recap what a DQN is.
# # #
# # # DQN algorithm
# # # -------------
# # #
# # # Our environment is deterministic, so all equations presented here are
# # # also formulated deterministically for the sake of simplicity. In the
# # # reinforcement learning literature, they would also contain expectations
# # # over stochastic transitions in the environment.
# # #
# # # Our aim will be to train a policy that tries to maximize the discounted,
# # # cumulative reward
# # # :math:`R_{t_0} = \sum_{t=t_0}^{\infty} \gamma^{t - t_0} r_t`, where
# # # :math:`R_{t_0}` is also known as the *return*. The discount,
# # # :math:`\gamma`, should be a constant between :math:`0` and :math:`1`
# # # that ensures the sum converges. It makes rewards from the uncertain far
# # # future less important for our agent than the ones in the near future
# # # that it can be fairly confident about.
# # #
# # # The main idea behind Q-learning is that if we had a function
# # # :math:`Q^*: State \times Action \rightarrow \mathbb{R}`, that could tell
# # # us what our return would be, if we were to take an action in a given
# # # state, then we could easily construct a policy that maximizes our
# # # rewards:
# # #
# # # .. math:: \pi^*(s) = \arg\!\max_a \ Q^*(s, a)
# # #
# # # However, we don't know everything about the world, so we don't have
# # # access to :math:`Q^*`. But, since neural networks are universal function
# # # approximators, we can simply create one and train it to resemble
# # # :math:`Q^*`.
# # #
# # # For our training update rule, we'll use a fact that every :math:`Q`
# # # function for some policy obeys the Bellman equation:
# # #
# # # .. math:: Q^{\pi}(s, a) = r + \gamma Q^{\pi}(s', \pi(s'))
# # #
# # # The difference between the two sides of the equality is known as the
# # # temporal difference error, :math:`\delta`:
# # #
# # # .. math:: \delta = Q(s, a) - (r + \gamma \max_a Q(s', a))
# # #
# # # To minimise this error, we will use the `Huber
# # # loss <https://en.wikipedia.org/wiki/Huber_loss>`__. The Huber loss acts
# # # like the mean squared error when the error is small, but like the mean
# # # absolute error when the error is large - this makes it more robust to
# # # outliers when the estimates of :math:`Q` are very noisy. We calculate
# # # this over a batch of transitions, :math:`B`, sampled from the replay
# # # memory:
# # #
# # # .. math::
# # #
# # #    \mathcal{L} = \frac{1}{|B|}\sum_{(s, a, s', r) \ \in \ B} \mathcal{L}(\delta)
# # #
# # # .. math::
# # #
# # #    \text{where} \quad \mathcal{L}(\delta) = \begin{cases}
# # #      \frac{1}{2}{\delta^2}  & \text{for } |\delta| \le 1, \\
# # #      |\delta| - \frac{1}{2} & \text{otherwise.}
# # #    \end{cases}
# # #
# # # Q-network
# # # ^^^^^^^^^
# # #
# # # Our model will be a convolutional neural network that takes in the
# # # difference between the current and previous screen patches. It has two
# # # outputs, representing :math:`Q(s, \mathrm{left})` and
# # # :math:`Q(s, \mathrm{right})` (where :math:`s` is the input to the
# # # network). In effect, the network is trying to predict the *quality* of
# # # taking each action given the current input.
# # #
# #
# # class DQN(nn.Module):
# #
# #     def __init__(self):
# #         super(DQN, self).__init__()
# #         self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
# #         self.bn1 = nn.BatchNorm2d(16)
# #         self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
# #         self.bn2 = nn.BatchNorm2d(32)
# #         self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
# #         self.bn3 = nn.BatchNorm2d(32)
# #         self.head = nn.Linear(448, 2)
# #
# #     def forward(self, x):
# #         x = F.relu(self.bn1(self.conv1(x)))
# #         x = F.relu(self.bn2(self.conv2(x)))
# #         x = F.relu(self.bn3(self.conv3(x)))
# #         return self.head(x.view(x.size(0), -1))
# #
# #
# # ######################################################################
# # # Input extraction
# # # ^^^^^^^^^^^^^^^^
# # #
# # # The code below are utilities for extracting and processing rendered
# # # images from the environment. It uses the ``torchvision`` package, which
# # # makes it easy to compose image transforms. Once you run the cell it will
# # # display an example patch that it extracted.
# # #
# #
# # resize = T.Compose([T.ToPILImage(),
# #                     T.Resize(40, interpolation=Image.CUBIC),
# #                     T.ToTensor()])
# #
# # # This is based on the code from gym.
# # screen_width = 600
# #
# #
# # def get_cart_location():
# #     world_width = env.x_threshold * 2
# #     scale = screen_width / world_width
# #     return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART
# #
# #
# # def get_screen():
# #     screen = env.render(mode='rgb_array').transpose(
# #         (2, 0, 1))  # transpose into torch order (CHW)
# #     # Strip off the top and bottom of the screen
# #     screen = screen[:, 160:320]
# #     view_width = 320
# #     cart_location = get_cart_location()
# #     if cart_location < view_width // 2:
# #         slice_range = slice(view_width)
# #     elif cart_location > (screen_width - view_width // 2):
# #         slice_range = slice(-view_width, None)
# #     else:
# #         slice_range = slice(cart_location - view_width // 2,
# #                             cart_location + view_width // 2)
# #     # Strip off the edges, so that we have a square image centered on a cart
# #     screen = screen[:, :, slice_range]
# #     # Convert to float, rescare, convert to torch tensor
# #     # (this doesn't require a copy)
# #     screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
# #     screen = torch.from_numpy(screen)
# #     # Resize, and add a batch dimension (BCHW)
# #     return resize(screen).unsqueeze(0).to(device)
# #
# #
# # env.reset()
# # plt.figure()
# # plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
# #            interpolation='none')
# # plt.title('Example extracted screen')
# # plt.show()
# #
# #
# # ######################################################################
# # # Training
# # # --------
# # #
# # # Hyperparameters and utilities
# # # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# # # This cell instantiates our model and its optimizer, and defines some
# # # utilities:
# # #
# # # -  ``select_action`` - will select an action accordingly to an epsilon
# # #    greedy policy. Simply put, we'll sometimes use our model for choosing
# # #    the action, and sometimes we'll just sample one uniformly. The
# # #    probability of choosing a random action will start at ``EPS_START``
# # #    and will decay exponentially towards ``EPS_END``. ``EPS_DECAY``
# # #    controls the rate of the decay.
# # # -  ``plot_durations`` - a helper for plotting the durations of episodes,
# # #    along with an average over the last 100 episodes (the measure used in
# # #    the official evaluations). The plot will be underneath the cell
# # #    containing the main training loop, and will update after every
# # #    episode.
# # #
# #
# # BATCH_SIZE = 128
# # GAMMA = 0.999
# # EPS_START = 0.9
# # EPS_END = 0.05
# # EPS_DECAY = 200
# # TARGET_UPDATE = 10
# #
# # policy_net = DQN().to(device)
# # target_net = DQN().to(device)
# # target_net.load_state_dict(policy_net.state_dict())
# # target_net.eval()
# #
# # optimizer = optim.RMSprop(policy_net.parameters())
# # memory = ReplayMemory(10000)
# #
# #
# # steps_done = 0
# #
# #
# # def select_action(state):
# #     global steps_done
# #     sample = random.random()
# #     eps_threshold = EPS_END + (EPS_START - EPS_END) * \
# #         math.exp(-1. * steps_done / EPS_DECAY)
# #     steps_done += 1
# #     if sample > eps_threshold:
# #         with torch.no_grad():
# #             return policy_net(state).max(1)[1].view(1, 1)
# #     else:
# #         return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)
# #
# #
# # episode_durations = []
# #
# #
# # def plot_durations():
# #     plt.figure(2)
# #     plt.clf()
# #     durations_t = torch.tensor(episode_durations, dtype=torch.float)
# #     plt.title('Training...')
# #     plt.xlabel('Episode')
# #     plt.ylabel('Duration')
# #     plt.plot(durations_t.numpy())
# #     # Take 100 episode averages and plot them too
# #     if len(durations_t) >= 100:
# #         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
# #         means = torch.cat((torch.zeros(99), means))
# #         plt.plot(means.numpy())
# #
# #     plt.pause(0.001)  # pause a bit so that plots are updated
# #     if is_ipython:
# #         display.clear_output(wait=True)
# #         display.display(plt.gcf())
# #
# #
# # ######################################################################
# # # Training loop
# # # ^^^^^^^^^^^^^
# # #
# # # Finally, the code for training our model.
# # #
# # # Here, you can find an ``optimize_model`` function that performs a
# # # single step of the optimization. It first samples a batch, concatenates
# # # all the tensors into a single one, computes :math:`Q(s_t, a_t)` and
# # # :math:`V(s_{t+1}) = \max_a Q(s_{t+1}, a)`, and combines them into our
# # # loss. By defition we set :math:`V(s) = 0` if :math:`s` is a terminal
# # # state. We also use a target network to compute :math:`V(s_{t+1})` for
# # # added stability. The target network has its weights kept frozen most of
# # # the time, but is updated with the policy network's weights every so often.
# # # This is usually a set number of steps but we shall use episodes for
# # # simplicity.
# # #
# #
# # def optimize_model():
# #     if len(memory) < BATCH_SIZE:
# #         return
# #     transitions = memory.sample(BATCH_SIZE)
# #     # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
# #     # detailed explanation).
# #     batch = Transition(*zip(*transitions))
# #
# #     # Compute a mask of non-final states and concatenate the batch elements
# #     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
# #                                           batch.next_state)), device=device, dtype=torch.uint8)
# #     non_final_next_states = torch.cat([s for s in batch.next_state
# #                                                 if s is not None])
# #     state_batch = torch.cat(batch.state)
# #     action_batch = torch.cat(batch.action)
# #     reward_batch = torch.cat(batch.reward)
# #
# #     # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
# #     # columns of actions taken
# #     state_action_values = policy_net(state_batch).gather(1, action_batch)
# #
# #     # Compute V(s_{t+1}) for all next states.
# #     next_state_values = torch.zeros(BATCH_SIZE, device=device)
# #     next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
# #     # Compute the expected Q values
# #     expected_state_action_values = (next_state_values * GAMMA) + reward_batch
# #
# #     # Compute Huber loss
# #     loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
# #
# #     # Optimize the model
# #     optimizer.zero_grad()
# #     loss.backward()
# #     for param in policy_net.parameters():
# #         param.grad.data.clamp_(-1, 1)
# #     optimizer.step()
# #
# #
# # ######################################################################
# # #
# # # Below, you can find the main training loop. At the beginning we reset
# # # the environment and initialize the ``state`` Tensor. Then, we sample
# # # an action, execute it, observe the next screen and the reward (always
# # # 1), and optimize our model once. When the episode ends (our model
# # # fails), we restart the loop.
# # #
# # # Below, `num_episodes` is set small. You should download
# # # the notebook and run lot more epsiodes.
# # #
# #
# # num_episodes = 50
# # for i_episode in range(num_episodes):
# #     # Initialize the environment and state
# #     env.reset()
# #     last_screen = get_screen()
# #     current_screen = get_screen()
# #     state = current_screen - last_screen
# #     for t in count():
# #         # Select and perform an action
# #         action = select_action(state)
# #         _, reward, done, _ = env.step(action.item())
# #         reward = torch.tensor([reward], device=device)
# #
# #         # Observe new state
# #         last_screen = current_screen
# #         current_screen = get_screen()
# #         if not done:
# #             next_state = current_screen - last_screen
# #         else:
# #             next_state = None
# #
# #         # Store the transition in memory
# #         memory.push(state, action, next_state, reward)
# #
# #         # Move to the next state
# #         state = next_state
# #
# #         # Perform one step of the optimization (on the target network)
# #         optimize_model()
# #         if done:
# #             episode_durations.append(t + 1)
# #             plot_durations()
# #             break
# #     # Update the target network
# #     if i_episode % TARGET_UPDATE == 0:
# #         target_net.load_state_dict(policy_net.state_dict())
# #
# # print('Complete')
# # env.render()
# # env.close()
# # plt.ioff()
# # plt.show()
