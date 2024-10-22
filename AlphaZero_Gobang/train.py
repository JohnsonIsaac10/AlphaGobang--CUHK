# -*- coding: utf-8 -*-

from __future__ import print_function

import heapq
import random
import numpy as np
from collections import defaultdict, deque
from Game import Board, Game
from MCTS_pure import MCTSPlayer as MCTS_Pure
from MCTS_Alpha import MCTSPlayer
from Network import PolicyValueNet
from torch.utils.tensorboard import SummaryWriter
from tensorboard import notebook
import pickle

class TrainPipeline():
    def __init__(self, init_model=None, init_data_buffer=None):
        # params of the board and the game
        self.board_width = 8
        self.board_height = 8
        self.n_in_row = 5
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 600  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        # self.data_buffer = deque(maxlen=self.buffer_size)
        self.data_buffer = pickle.load(file=open('new_data_buffer', 'rb'))
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 3000
        self.best_win_ratio = 0.0
        self.first_player = []
        self.winner = []
        self.avg_episode_len = []
        # self.weights_list = deque(maxlen=self.buffer_size)
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000

        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)
        self.writer = SummaryWriter('./new_event/tensorboard')

    def get_equi_data(self, play_data):
        extend_data = []
        for state, mcts_porb, winner, weight in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner, weight))
                # extend_weights.extend(weights)
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner, weight))
                # extend_weights.extend(weights)
        return extend_data

    def construct_weights(self, epsisode_len, gamma=0.95):
        w = np.empty((int(epsisode_len),), np.float32)
        w[epsisode_len - 1] = 1.0  # 最靠后的权重最大
        for i in range(epsisode_len - 2, -1, -1):
            w[i] = w[i + 1] * gamma
        return epsisode_len * w / np.sum(w)  # 所有元素之和为length

    def a_res(self, samples, weights, m):
        """
        :samples: [(item, weight), ...]
        :k: number of selected items
        :returns: [(item, weight), ...]
        """
        heap = []  # [(new_weight, item), ...]
        for sample, wi in zip(samples, weights):
            ui = random.uniform(0, 1)
            ki = ui ** (1 / wi)

            if len(heap) < m:
                heapq.heappush(heap, (ki, sample))
            elif ki > heap[0][0]:
                heapq.heappush(heap, (ki, sample))

                if len(heap) > m:
                    heapq.heappop(heap)

        return [item[1] for item in heap]

    def collect_selfplay_data(self, n_games=1):
        for i in range(n_games):
            first_player, winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            if random.random() < -0.0812 * self.episode_len + 1.6364:
                print('drop play data')
                break

            # augment the data
            # weights = self.construct_weights(epsisode_len=self.episode_len)
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)
            # self.data_buffer.extend(weights)
            self.winner.append(winner)
            self.avg_episode_len.append(self.episode_len)

            if winner == 1:
                if random.random() < self.winner.count(2) / (self.winner.count(1)) - 1:
                    self.data_buffer.extend(play_data)
                    self.winner.append(winner)
                    self.avg_episode_len.append(self.episode_len)
            elif winner == 2:
                if random.random() < self.winner.count(1) / (self.winner.count(2) * 0.8) - 1:
                    self.data_buffer.extend(play_data)
                    self.winner.append(winner)
                    self.avg_episode_len.append(self.episode_len)

            print('winner 1: ', self.winner.count(1))
            print('winner 2: ', self.winner.count(2))
            print('avg len: ', sum(self.avg_episode_len)/len(self.avg_episode_len))

    def policy_update(self):
        """update the policy-value net"""
        weights = [data[3] for data in self.data_buffer]
        # mini_batch = random.choices(self.data_buffer, cum_weights=weights, k=self.batch_size)
        mini_batch = self.a_res(self.data_buffer, weights, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy, value_loss = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy, value_loss

    def policy_evaluate(self, n_games=10):

        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("number of playouts:{}, win: {}, lose: {}, tie:{}, win ratio:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1], win_ratio))
        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:
            notebook.start("--logdir ./new_event/tensorboard")
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))
                self.writer.add_scalar('episode length', self.episode_len, i)
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy, value_loss = self.policy_update()
                    self.writer.add_scalar('loss', loss, i)
                    self.writer.add_scalar('entropy', entropy, i)
                    self.writer.add_scalar('value loss', value_loss, i)
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model('./current_model_8_8_600playouts_after3500.model')
                    pickle.dump(self.data_buffer, open('new_data_buffer_after3500', 'wb'))
                    print('Data store finish')
                    if win_ratio > self.best_win_ratio:
                        print("New best policy")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model('./best_model_8_8_600playouts_after3500.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = TrainPipeline(init_model='current_model_8_8_600playouts.model')
    training_pipeline.run()
    # notebook.start("--logdir ./data/tensorboard")
    notebook.start("--logdir ./new_event/tensorboard")


