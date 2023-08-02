import os
# import pickle
import random
import numpy as np
import torch
from collections import defaultdict
from fedscale.utils.compressors.quantization import QSGDCompressor
import sys
import logging
import time
import pickle
from fedscale.cloud import commons
# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)


class RL:
    def __init__(self, total_clients=200, participation_rate=20, discount_factor=0.5, learning_rate=0.01, exploration_prob=0.5, actions=[1, 2]):
        self.total_clients = total_clients
        self.participation_rate = participation_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.actions = commons.ACTIONS
        self.Q = {i: {} for i in range(self.total_clients)}
        self.max_q_table_size = 10000
        self.overhead_times = defaultdict(float)  # Track overhead times

    def choose_action_per_client(self, global_state, local_state, selected_client):
        try:
            prob = np.random.rand()
            logging.info("prob: {}".format(prob))
            state_key = tuple(global_state.items()), tuple(local_state.items())  # Convert dictionaries to tuples
            if prob < self.exploration_prob:
                action = np.random.choice(self.actions)
                logging.info("Exploration: client {} chooses action {}".format(selected_client, action))
            else:
                start_time = time.time()
                q_values = self.Q.get(selected_client).get(state_key)
                logging.info("q_values: {}".format(q_values))
                if q_values:
                    # Separate rewards for each objective
                    objective_rewards = {objective: q_values[objective] for objective in q_values}
                    logging.info("objective_rewards: {}".format(objective_rewards))
                    # Choose action based on the rewards for each objective
                    # action = max(objective_rewards, key=objective_rewards.get)
                    # Calculate the combined score for each entry
                    max_combined_score = 0.0
                    action = None

                    # Iterate through the data to find the key with the maximum combined score
                    for key, value in objective_rewards.items():
                        combined_score = value['participation_success'] + value['accuracy']
                        if combined_score > max_combined_score:
                            max_combined_score = combined_score
                            action = key
                    if action is None:
                        action = np.random.choice(self.actions)
                    logging.info("Exploitation: client {} chooses action {}".format(selected_client, action))
                
                else:
                    print("State is still unexplored, Cannot exploit!")
                    action = np.random.choice(self.actions)
                
                end_time = time.time()
                overhead_time = end_time - start_time
                logging.info(f'Overhead time for choose_action_per_client: {overhead_time} seconds')
                self.overhead_times['choose_action_per_client'] += overhead_time
        except Exception as e:
            logging.error(f"Error in choose_action_per_client {e}")
            action = np.random.choice(self.actions)
        return action

    
    def update_Q_per_client(self, selected_client, global_state, local_state, action, new_global_state, new_local_state, rewards):
        try:
            start_time = time.time()
            client_Q = self.Q[selected_client]
            state_key = tuple(global_state.items()), tuple(local_state.items())  # Convert dictionaries to tuples
            new_state_key = tuple(new_global_state.items()), tuple(new_local_state.items())  # Convert dictionaries to tuples

            # Initialize Q-values for state_key and action if they don't exist
            if state_key not in client_Q:
                client_Q[state_key] = {}
            if action not in client_Q[state_key]:
                client_Q[state_key][action] = {objective: 0 for objective in rewards}

            Q_current = client_Q[state_key][action]
            Q_future = max(client_Q[new_state_key].values(), key=lambda item: (item['participation_success'], item['accuracy']))

            logging.info(f'Faraz - debug before normalization: Q_current: {client_Q[new_state_key].values()}')
            # Normalize Q_future values for 'participation_success' and 'accuracy'
            max_participation_success = max(item['participation_success'] for item in client_Q[state_key].values())
            max_accuracy = max(item['accuracy'] for item in client_Q[state_key].values())
            for item in client_Q[new_state_key].values():
                # if max_participation_success == 0:
                #     item['participation_success'] = 0
                # else:
                #     item['participation_success'] /= max_participation_success
                if max_accuracy == 0:
                    item['accuracy'] = 0
                else:
                    item['accuracy'] /= max_accuracy
            logging.info(f'Faraz - debug after normalization: Q_current: {client_Q[new_state_key].values()}')
                
            
            # Update Q-values separately for each objective
            for objective, reward in rewards.items():
                updated_Q = Q_current[objective] + self.learning_rate * (reward + self.discount_factor * Q_future[objective] - Q_current[objective])
                client_Q[state_key][action][objective] = updated_Q

            # self.limit_q_table_size(client_Q)
            end_time = time.time()
            overhead_time = end_time - start_time
            self.overhead_times['update_Q_per_client'] += overhead_time
            logging.info(f'Overhead time for update_Q_per_client: {overhead_time} seconds')
            logging.info("Updated Q for client {} with state_key {}, action {}, and values {}".format(selected_client, state_key, action, self.Q[state_key][action]))
        except Exception as e:
            logging.error(f"Error in update_Q_per_client {e}")
            
    
    def limit_q_table_size(self, q_table):
        try:
            if len(q_table) > self.max_q_table_size:
                state_action_pairs = sorted(q_table.keys(), key=lambda x: random.random())[:self.max_q_table_size]
                q_table.clear()
                for state, action in state_action_pairs:
                    action = np.random.choice(self.actions)
                    q_table[state][action] = action
        except Exception as e:
            logging.error("Error in limit_q_table_size")
            logging.error(e)

    def save_Q(self, path):
        try:
            path = os.path.join(path, 'Q.pkl')
            with open(path, 'wb') as f:
                pickle.dump(self.Q, f)
        except Exception as e:
            logging.error("Error in save_Q")
            logging.error(e)

    def load_Q(self, path):
        try:
            path = os.path.join(path, 'Q.pkl')
            logging.info("Loading Q from {}".format(path))
            with open(path, 'rb') as f:
                self.Q = pickle.load(f)
        except Exception as e:
            logging.error("Error in load_Q")
            logging.error(e)

    def print_overhead_times(self):
        try:
            for method, time_taken in self.overhead_times.items():
                logging.info(f"Total overhead time for {method}: {time_taken} seconds")
        except Exception as e:
            logging.error("Error in print_overhead_times")
            logging.error(e)



