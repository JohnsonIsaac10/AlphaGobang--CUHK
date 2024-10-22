
from __future__ import print_function
import numpy as np


class Board(object):
    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        square_state = np.zeros((4, self.width, self.height))
        if len(self.states):
            moves, players = np.array(list(zip(*self.states.items())))
            # current player's move
            move_curr = moves[players == self.current_player]
            move_prev = moves[0:len(move_curr)-1]
            # opponent player's move
            move_oppo = moves[players != self.current_player]
            move_oppo_prev = move_oppo[0:len(move_oppo)-1]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # square_state[2][move_prev // self.width,
            #                 move_prev % self.height] = 1.0
            # square_state[3][move_oppo_prev // self.width,
            #                 move_oppo_prev % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    def __init__(self, board, **kwargs):
        self.board = board
        self.board_origin = Board(width=board.width, height=board.height, n_in_row=board.n_in_row)


    def graphic(self, board, player1, player2):
        width = board.width
        height = board.height

        print("Player", player1, "with ●".rjust(3))
        print("Player", player2, "with ○".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('●'.center(8), end='')
                elif p == player2:
                    print('○'.center(8), end='')
                else:
                    print('-'.center(8), end='')
            print('\r\n\r\n')

    def construct_weights(self, epsisode_len, gamma=0.95):
        w = np.empty((int(epsisode_len),), np.float32)
        w[epsisode_len - 1] = 1.0
        for i in range(epsisode_len - 2, -1, -1):
            w[i] = w[i + 1] * gamma
        return epsisode_len * w / np.sum(w)

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                episode_len = len(current_players)
                weights = self.construct_weights(episode_len)
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return current_players[0], winner, zip(states, mcts_probs, winners_z, weights)

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        move_list = []
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            move_list.append(int(move))
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner, move_list

    def start_play_battle(self, player1, player2, start_player=0, is_shown=1):
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        self.board_origin.init_board(int(start_player/1))
        # test = int(start_player / 1)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        episode = 0
        move_list = []
        if is_shown:
            self.graphic(self.board_origin, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            # print(current_player)
            if current_player == 1:
                player_in_turn = players[current_player]
                if episode == 0:
                    move = np.random.choice(self.board.availables)
                else:
                    move = player_in_turn.get_action(self.board_origin)

            else:
                player_in_turn = players[current_player]
                if episode == 0:
                    move = np.random.choice(self.board.availables)
                else:
                    move = player_in_turn.get_action(self.board)
            # print(move)
            move_list.append(int(move))
            self.board.do_move(move)
            self.board_origin.do_move(move)
            episode += 1

            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:

                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                        print(winner)
                    else:
                        print("Game end. Tie")
                return winner, move_list
