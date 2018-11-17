# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import game
from pystockfish import *
deep = Engine(depth=20)

###############################################################################
# Piece-Square tables. Tune these to change sunfish's behaviour
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

###############################################################################
# Global constants
###############################################################################

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





EPISODES = 1000

class DQNAgent:
    def __init__(self, state_size=18, action_size=1889):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

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
        print (len(valid_move_indices))
        if np.random.rand() <= self.epsilon:
            random_action_index = random.randrange(self.action_size)
            while random_action_index not in valid_move_indices:
                random_action_index = random.randrange(self.action_size)
            return random_action_index
        act_values = self.model.predict(state)
        new_act_values = []
        for i,val in enumerate(act_values[0]):
            if i in valid_move_indices:
                new_act_values.append(val)
            else:
                new_act_values.append(0.0)
        #valid_act_values = [act_values[i] for i in valid_move_indices]
        #print("returned action", np.argmax(valid_act_values[0]))
        return np.argmax(new_act_values)  # returns action

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

if __name__ == "__main__":
    #env = gym.make('CartPole-v1')
    state_size = 18 #env.observation_space.shape[0]
    # print(state_size)
    action_size = 1889 #env.action_space.n
    # print(action_size)
    agent = DQNAgent() # call this in game
    #agent.load("/save/cartpole-dqn.h5")
    batch_size = 32

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

    for e in range(EPISODES):
        done = False
        print ("episode: ", e)

        # a game has started
        pos = game.Position(initial, 0, (True,True), (True,True), 0, 0) # game.py stuff
         # game.py stuff
        searcher = game.Searcher() # game.py stuff
        moves_list = [] # game.py stuff

        round = 0

        while True:
            round += 1
            game.print_pos(pos)
            rawState = pos
            state = toBit(rawState)
            #state = np.reshape(state, [1, state_size])
            # just evaluating board
            before_output = deep.bestmove()
            score_before_model_move = int(before_output['info'].split(" ")[9])
            if pos.score <= -MATE_LOWER:
                print("You lost")
                break
            # asking DQN agent for action
            move = None
            valid_moves = [m for m in pos.gen_moves()] #### (85,65), (87, 97)
            valid_move_indices = [possible_actions.index(gm) for gm in valid_moves]
            # coordinates
            dqn_move_index = agent.act(state, valid_move_indices) ## returns index of maximum value action
            if dqn_move_index not in valid_move_indices:
                print("made invalid move")
                break
            dqn_move = possible_actions[dqn_move_index]

            pos = pos.move(dqn_move)
            dqn_move_stockfish = game.render(119-dqn_move[0]) + game.render(119-dqn_move[1])
            moves_list.append(dqn_move_stockfish)
            deep.setposition(moves_list)
            after_output = deep.bestmove()
            score_after_model_move = int(after_output['info'].split(" ")[9])

            ## Q LEARNING PART
            ########## next_state, reward, done, _ = env.step(action) ####### GET REWARD
            #reward = reward if not done else -10
            new_state = toBit(pos.getNewState(dqn_move))
            #new_state = np.reshape(new_state, [1, state_size])
            reward = score_after_model_move - score_before_model_move
            agent.remember(state, dqn_move_index, reward, new_state, done)
            state = new_state
            game.print_pos(pos.rotate())

            if pos.score <= -MATE_LOWER:
                print("You won")
                done = True

            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, round, agent.epsilon))
                break

            # OPPONENT MOVE
            opponent_move, score = searcher.search(pos, secs=2)

            opponent_move_stockfish = game.render(119-opponent_move[0]) + game.render(119-opponent_move[1])
            pos = pos.move(opponent_move)
            moves_list.append(opponent_move_stockfish)
            deep.setposition(moves_list)
            #game.print_pos(pos.rotate())

            if score == MATE_UPPER:
                print("Checkmate!")
                done = True

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, round, agent.epsilon))
                break

        #     ########## OLD CODE ###########
        # for time in range(500):
        #     env.render()
        #     action = agent.act(state) ## implement agent.act
        #     if e == 0:
        #         print (action)
        #     next_state, reward, done, _ =env.step(action)
        #     reward = reward if not done else -10
        #     next_state = np.reshape(next_state, [1, state_size])
        #     agent.remember(state, action, reward, next_state, done)
        #     state = next_state
        # # if e % 10 == 0:
        # #     agent.save("/save/cartpole-dqn.h5")
