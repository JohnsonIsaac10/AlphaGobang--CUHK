
from __future__ import print_function
from Game import Board, Game
from MCTS_Alpha import MCTSPlayer

from Network import PolicyValueNet
import sys
import json

from PyQt5.QtWidgets import QApplication

from View.flat_chess_board_interface import FlatChessBoardInterface

class Human(object):
    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def save_board(file_path):
    # 读入棋谱
    with open(file_path, 'r', encoding='utf-8') as f:
        games = json.load(f)

    # 创建界面
    app = QApplication(sys.argv)
    window = FlatChessBoardInterface()
    window.show()

    # 绘制棋谱
    for i, game in enumerate(games, 1):
        window.clearBoard()
        window.drawGame(game)
        window.save(fr'human_play/new-play2.png')

    sys.exit(app.exec_())


def run():
    n = 5
    width, height = 8, 8
    model_file = 'current_model_8_8_600playouts_at5500.model'

    # try:
    board = Board(width=width, height=height, n_in_row=n)
    game = Game(board)

    best_policy = PolicyValueNet(width, height, model_file = model_file)
    mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=600)
    human = Human()

    # set start_player=0 for human first
    start_player = int(input('Who plays first\n'
                             '0: human first\n'
                             '1: machine first\n'))
    actions_list = []
    winner, move_list = game.start_play(human, mcts_player, start_player=start_player,
                                           is_shown=1)
    actions_list.append(move_list)
    file_path = './human_play/human_play_game.json'
    json.dump(actions_list, open(file_path, 'w'))
    save_board(file_path)


if __name__ == '__main__':
    run()
