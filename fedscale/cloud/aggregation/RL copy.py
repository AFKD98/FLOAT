import os
import pickle
import random
import numpy as np
import torch
from collections import defaultdict
from fedscale.utils.compressors.quantization import QSGDCompressor
import sys
import logging
import time

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)


class RL:
    def __init__(self, total_clients=200, participation_rate=20, discount_factor=0.5, learning_rate=0.01, exploration_prob=0.5, q_bits=[8, 16]):
        self.total_clients = total_clients
        self.participation_rate = participation_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.q_bits = q_bits
        self.Q = {i: defaultdict(lambda: defaultdict(int)) for i in range(self.total_clients)}
        self.max_q_table_size = 10000
        self.overhead_times = defaultdict(float)  # Track overhead times

    def choose_action_per_client(self, global_state, local_state, selected_client):
        try:
            prob = np.random.rand()
            logging.info("prob: {}".format(prob))
            state_key = tuple(global_state.items()), tuple(local_state.items())  # Convert dictionaries to tuples
            if prob < self.exploration_prob:
                action = np.random.choice(self.q_bits)
                logging.info("Exploration: client {} chooses action {}".format(selected_client, action))
            else:
                start_time = time.time()
                q_values = self.Q.get(selected_client).get(state_key)
                # logging.info(f'Faraz - debug np.argmax(list(q_values.keys())): {np.argmax(list(q_values.keys()))}')
                # logging.info(f'Faraz - debug list(q_values): {q_values}')
                # logging.info(f'Faraz - debug list(q_values): {q_values.values()}')
                max_q = np.argmax(list(q_values.values()))
                action = int(list(q_values.keys())[max_q])
                # logging.info(f'Faraz - debug action: {action}')
                if action == 0:
                    action = np.random.choice(self.q_bits)
                # logging.info("Exploitation: client {} chooses action {}".format(selected_client, action))
                end_time = time.time()
                overhead_time = end_time - start_time
                logging.info(f'Overhead time for choose_action_per_client: {overhead_time} seconds')
                self.overhead_times['choose_action_per_client'] += overhead_time
        except:
            logging.error("Error in choose_action_per_client")
            action = np.random.choice(self.q_bits)
        return action
    
    def update_Q_per_client(self, selected_client, global_state, local_state, action, new_global_state, new_local_state, reward):
        start_time = time.time()
        client_Q = self.Q[selected_client]
        state_key = tuple(global_state.items()), tuple(local_state.items())  # Convert dictionaries to tuples
        new_state_key = tuple(new_global_state.items()), tuple(new_local_state.items())  # Convert dictionaries to tuples
        Q_current = client_Q[state_key][action]
        Q_future = np.max(list(client_Q[new_state_key].values()))
        updated_Q = Q_current + self.learning_rate * (reward + self.discount_factor * Q_future - Q_current)
        client_Q[state_key][action] = updated_Q
        self.limit_q_table_size(client_Q)
        end_time = time.time()
        overhead_time = end_time - start_time
        self.overhead_times['update_Q_per_client'] += overhead_time
        logging.info(f'Overhead time for update_Q_per_client: {overhead_time} seconds')
        logging.info("Updated Q for client {} with state_key {}, action {}, and value {}".format(selected_client, state_key, action, updated_Q))
    
    def limit_q_table_size(self, q_table):
        if len(q_table) > self.max_q_table_size:
            state_action_pairs = sorted(q_table.keys(), key=lambda x: random.random())[:self.max_q_table_size]
            q_table.clear()
            for state, action in state_action_pairs:
                q_table[state][action] = np.random.choice(self.q_bits)

    def save_Q(self, path):
        try:
            logging.info("Saving Q to {}".format(path))
            with open(path, 'wb') as f:
                pickle.dump(self.Q, f)
        except Exception as e:
            logging.error("Error in save_Q")
            logging.error(e)

    def load_Q(self, path):
        with open(path, 'rb') as f:
            self.Q = pickle.load(f)

    def print_overhead_times(self):
        for method, time_taken in self.overhead_times.items():
            logging.info(f"Total overhead time for {method}: {time_taken} seconds")

        # logging.info('self.Q: {}'.format(self.Q))
    def limit_q_table_size(self, q_table):
        if len(q_table) > self.max_q_table_size:
            state_action_pairs = sorted(q_table.keys(), key=lambda x: random.random())[:self.max_q_table_size]
            q_table.clear()
            for state_action in state_action_pairs:
                q_table[state_action] = 0


