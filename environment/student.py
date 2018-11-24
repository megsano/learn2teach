


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
import game
from pystockfish import *
import ray
from ray import tune

ray.init(ignore_reinit_error=True)

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

class DQNAgent:
    def __init__(self, state_size=18, action_size=1856):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = params['discount']    # discount rate
        self.epsilon = params['epsilon']# exploration rate
        self.epsilon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']
        self.learning_rate = params['lr']
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(params['first_neuron'], input_dim=self.state_size, activation=params['activation1']))
        model.add(Dense(params['second_neuron'], activation=params['activation2']))
        model.add(Dense(self.action_size, activation=params['activation3']))
        model.compile(loss=params['losses'],
                      optimizer=params['optimizer'])
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
            if not done:
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
    splitted = []
    for i in range(8):
        splitted.append(boardString[8 * i : 8 * i + 8])
    for l, line in enumerate(splitted): # "".join(board.split('\n')[:-1])
        #print(line)
        for col in range(8):
            #print (line[col])
            if line[col] == piece_letter:
                return (l, col)

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

def train(reporter):

    # Constants for training
    EPISODES = 100
    state_size = 18
    action_size = 1856
    agent = DQNAgent()
    batch_size = params['batch_size']

    # Creating a list of all possible actions on the chessboard
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
    assert len(possible_actions) == action_size

    scores_list = []

    # Training EPISODES times
    for e in range(EPISODES):
        ray.experimental.flush_redis_unsafe()
        ray.experimental.flush_task_and_object_metadata_unsafe()
        deep = Engine(depth=20)
        final_score = 0
        print ("episode: ", e)
        done = False
        pos = game.Position(initial, 0, (True,True), (True,True), 0, 0)
        searcher = game.Searcher()
        moves_list = []
        round = 0
        while True: # While no one has won, keep playing
            round += 1
            #print("dqn to move")
            #game.print_pos(pos)
            state = toBit(pos)
            # computing score of board before agent makes action
            before_output = deep.bestmove()
            #print ("before output info: " + before_output['info'])
            before_output_list = before_output['info'].split(" ")
            if 'mate' in before_output_list:
                mate_index = before_output_list.index('mate')
                mate_score = before_output_list[mate_index + 1]
                score_before_model_move = -5000 #/ float(math.sqrt(float(mate_score)))
                print("mate is in before_output_list so score is ", str(score_before_model_move))
            else:
                score_before_model_move = (-1)*int(before_output['info'].split(" ")[9])
            # if we can't find cp in it, or if we find mate in it
            # that means we are closed to being checkmated
            # score should be a function of number after 'mate'

            # if before_output['info'].split(" ")[8] == 'mate':
            #     print("oh gr8, m8 in ", before_output['info'].split(" ")[9])
            #     print("episode: {}/{}, number of rounds: {}, score: {}, e: {:.2}"
            #           .format(e, EPISODES, round, final_score / float(round), agent.epsilon))
            #     break


            possibly_valid_moves = [m for m in pos.gen_moves(False)]
            possibly_valid_move_indices = [possible_actions.index(gm) for gm in possibly_valid_moves]

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
                      .format(e, EPISODES, round, final_score / float(round), agent.epsilon))
                scores_list.append(final_score / float(round))
                print (scores_list)
                reporter(mean_accuracy=final_score)
                break
            # else:
            #     valid_move_indices = possibly_valid_move_indices

            ''' End check for check code'''

            # DQN chooses an action
            #valid_moves = [m for m in pos.gen_moves()] #### (85,65), (87, 97)
            #valid_move_indices = [possible_actions.index(gm) for gm in valid_moves]
            dqn_move_index = agent.act(state, valid_move_indices)
            # a check that we should be removing later
            while dqn_move_index not in valid_move_indices:
                dqn_move_index = random.choice(valid_move_indices)

            if dqn_move_index not in valid_move_indices:
                print("made invalid move")
                break

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
            moves_list.append(dqn_move_stockfish)
            #print("dqn move stockfish: ", str(dqn_move_stockfish))
            #print(moves_list)
            deep.setposition(moves_list)
            # compute score of board after agent makes action
            after_output = deep.bestmove()
            #print ("after output info: " + after_output['info'])
            after_output_list = after_output['info'].split(" ")
            if 'mate' in after_output_list:
                mate_index = after_output_list.index('mate')
                mate_score = after_output_list[mate_index + 1]
                score_after_model_move = -5000 #/ float(math.sqrt(float(mate_score)))
                print("mate is in after_output_list so score is ", str(score_after_model_move))
            else:
                score_after_model_move = (-1)*int(after_output['info'].split(" ")[9])

            # Q-Learning
            pos.rotate()
            new_state = toBit(pos.getNewState(dqn_move))
            reward = score_after_model_move - score_before_model_move #abs(score_after_model_move - score_before_model_move)*(score_after_model_move - score_before_model_move)
            #print(reward)
            final_score += reward
            agent.remember(state, dqn_move_index, reward, new_state, done)
            state = new_state

            pos.rotate()


            # # if checkmate happened, break out of the loop and print stats
            # if done:
            #     print("episode: {}/{}, score: {}, e: {:.2}"
            #           .format(e, EPISODES, round, agent.epsilon))
            #     break

            '''Begin check for check code'''

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
                      .format(e, EPISODES, round, final_score / float(round), agent.epsilon))
                scores_list.append(final_score / float(round))
                print (scores_list)
                reporter(mean_accuracy=final_score)
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
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

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
    agent.save('save/agent.h5')
        #     ########## OLD CODE ###########
        # # if e % 10 == 0:
        # #     agent.save("/save/cartpole-dqn.h5")


params = {'lr':lambda spec: np.random.uniform(0.1, 0.5),
    'first_neuron':lambda spec: np.random.choice(4, 8, 16, 32, 64, 128),
     'second_neuron':lambda spec: np.random.choice(4, 8, 16, 32, 64, 128),
     'hidden_layers':lambda spec: np.random.choice(2,3,4,5,6),
     'batch_size': lambda spec: np.random.choice(2,4,8,16,32),
     'discount': lambda spec: np.random.uniform(0.95, 0.99),
     'epsilon': lambda spec: np.random.uniform(0.3, 0.8),
     'epsilon_min': lambda spec: np.random.uniform(0.1, 0.3),
     'epsilon_decay': lambda spec: np.random.uniform(0.990, 0.999),
     #'epochs': [300],
     'dropout': lambda spec: np.random.uniform(0, 10)
     # 'weight_regulizer':[None],
     # 'emb_output_dims': [None],
     # 'optimizer': [Adam, Nadam],
     # 'losses': [categorical_crossentropy, logcosh],
     # 'activation':[relu, elu],
     # 'last_activation': [softmax]
     }

if __name__ == "__main__":
    configuration = tune.Experiment(
      "experiment1",
      run=train,
      stop = {"mean_accuracy" : 0},
      config = params
    )
