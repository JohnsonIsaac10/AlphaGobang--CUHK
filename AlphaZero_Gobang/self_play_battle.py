
from __future__ import print_function
from Game import Board, Game
from MCTS_Alpha import MCTSPlayer

from Network import PolicyValueNet
from MCTS_pure import MCTSPlayer as MCTS_Pure
import sys
import json

from PyQt5.QtWidgets import QApplication

from View.flat_chess_board_interface import FlatChessBoardInterface


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
        window.save(fr'self_play_battle/{i}.png')
    sys.exit(app.exec_())


def run():
    n = 5
    n_battle_games = 1
    width, height = 8, 8
    model_file = 'current_model_8_8_600playouts_at4500.model'
    model_file_origin = 'current_model_8_8_600playouts_at4500.model'

    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        best_policy = PolicyValueNet(width, height, model_file = model_file)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        best_policy_origin = PolicyValueNet(width, height, model_file=model_file_origin)
        mcts_player_origin = MCTSPlayer(best_policy_origin.policy_value_fn, c_puct=5, n_playout=400)

        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=3000)

        # set start_player=0 for human first
        start_player = 0
        p1, p2 = board.players
        players = {p1:0, p2: 0, -1: 0}
        state_actions = []
        for i in range(n_battle_games):
            winner, move_list = game.start_play_battle(mcts_player_origin, mcts_player, start_player=start_player, is_shown=1)
            players[winner] += 1
            print('winner is ', winner)
            state_actions.append(move_list)
        file_path = './self_play_battle/self_play_games.json'
        json.dump(state_actions, open(file_path, 'w'))
        save_board(file_path)
        print('player {} first'.format(start_player+1))
        print(players)

    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
