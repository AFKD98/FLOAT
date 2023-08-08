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
random.seed(1)
np.random.seed(2)


class RL:
    def __init__(self, total_clients=200, participation_rate=50, discount_factor=0, learning_rate=0.1, exploration_prob=1.0, actions=[1, 2]):
        self.total_clients = total_clients
        self.participation_rate = participation_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.actions = commons.ACTIONS
        path = '/home/ahmad/FedScale/benchmark/logs/rl_model/Q.pkl'
        if os.path.exists(path):
            logging.info("Loading Q from {}".format(path))
            with open(path, 'rb') as f:
                self.Q = pickle.load(f)
        else:
            self.Q = {}
        self.max_q_table_size = 10000
        self.overhead_times = defaultdict(float)  # Track overhead times

    def choose_random_action_uniformly(self, state_key):
        
        #select an action if it has previously not been selected
        random_action = np.random.choice(self.actions)
        unexplored_actions = [action for action in self.actions if action not in self.Q.get(state_key, {})]
        # logging.info(f'unexplored_actions: {unexplored_actions}')
        if len(unexplored_actions) > 0:
            random_action = np.random.choice(unexplored_actions)
            return random_action
        #choose the least count action if count for this action is greater than 5
        elif self.Q.get(state_key):
            actions_for_state = self.Q.get(state_key)
        
            if random_action in actions_for_state:
                # logging.info(f'actions dict: {actions_for_state.items()}')
                sorted_actions_for_state = sorted(actions_for_state.items(), key=lambda x: int(x[1]['count']))
                # logging.info(f'sorted_actions_for_state: {sorted_actions_for_state}')
                random_action = sorted_actions_for_state[0][0]
        else:
            random_action = np.random.choice(self.actions)
        return random_action
    
    def choose_action_per_client(self, global_state, local_state, selected_client):
        try:
            prob = np.random.rand()
            logging.info("prob: {}".format(prob))
            state_key = tuple(global_state.items()), tuple(local_state.items())  # Convert dictionaries to tuples
            if prob < self.exploration_prob:
                # action = np.random.choice(self.actions)
                action = self.choose_random_action_uniformly(state_key)
                logging.info("Exploration: client {} chooses action {}".format(selected_client, action))
            else:
                start_time = time.time()
                q_values = self.Q.get(state_key)
                logging.info("q_values: {}".format(q_values))
                if q_values:
                    # Separate rewards for each objective
                    action_rewards = {action: q_values[action] for action in q_values}
                    logging.info("action_rewards: {}".format(action_rewards))
                    # Choose action based on the rewards for each objective
                    # action = max(objective_rewards, key=objective_rewards.get)
                    # Calculate the combined score for each entry
                    max_combined_score = 0.0
                    action = None

                    # Iterate through the data to find the key with the maximum combined score
                    average_acc = sum([value['accuracy'] for key, value in action_rewards.items()]) / len(action_rewards)
                    if average_acc < 0.5:
                        w_p = 0.7
                        w_a = 0.3
                    else:
                        w_p = 0.3
                        w_a = 0.7

                    for key, value in action_rewards.items():
                        combined_score = w_p*value['participation_success'] + (w_a*value['accuracy'])/value['count']
                        if combined_score > max_combined_score:
                            max_combined_score = combined_score
                            action = key
                    if action is None:
                        # action = np.random.choice(self.actions)
                        action = self.choose_random_action_uniformly(state_key)
                    logging.info("Exploitation: client {} chooses action {}".format(selected_client, action))
                
                else:
                    print("State is still unexplored, Cannot exploit!")
                    # action = np.random.choice(self.actions)
                    action = self.choose_random_action_uniformly(state_key)
                
                end_time = time.time()
                overhead_time = end_time - start_time
                logging.info(f'Overhead time for choose_action_per_client: {overhead_time} seconds')
                self.overhead_times['choose_action_per_client'] += overhead_time
        except Exception as e:
            logging.error(f"Error in choose_action_per_client {e}")
            action = np.random.choice(self.actions)
            # action = self.choose_random_action_uniformly(state_key)
        return action

    
    def update_Q_per_client(self, selected_client, global_state, local_state, action, new_global_state, new_local_state, rewards, round=None):
        try:
            start_time = time.time()
            # client_Q = self.Q[selected_client]
            state_key = tuple(global_state.items()), tuple(local_state.items())  # Convert dictionaries to tuples
            new_state_key = tuple(new_global_state.items()), tuple(new_local_state.items())  # Convert dictionaries to tuples

            # logging.info(f'New state_key and action: {state_key}, {action}')
            # logging.info(f'existing Q: {self.Q}')
            # Initialize Q-values for state_key and action if they don't exist
            if state_key not in self.Q.keys():
                self.Q[state_key] = {}
                # logging.info(f'State key {state_key} not in Q, initializing')
            if action not in self.Q[state_key].keys():
                # logging.info(f'Action {action} not in Q, initializing')
                self.Q[state_key][action] = {objective: 0 for objective in rewards}
                self.Q[state_key][action]['count'] = 0

            Q_current = self.Q[state_key][action]
            Q_current['count'] += 1
            Q_future = max(self.Q[new_state_key].values(), key=lambda item: (item['participation_success'], item['accuracy']))

            # Normalize Q_future values for 'participation_success' and 'accuracy'
            # max_participation_success = max(item['participation_success'] for item in self.Q[state_key].values())
            # max_accuracy = max(item['accuracy'] for item in self.Q[state_key].values())
            # for item in self.Q[new_state_key].values():
            #     if max_participation_success == 0:
            #         item['participation_success'] = 0
            #     else:
            #         item['participation_success'] /= max_participation_success
            #     if max_accuracy == 0:
            #         item['accuracy'] = 0
            #     else:
            #         item['accuracy'] /= max_accuracy
            learning_rate = max(self.learning_rate*round/50, 1.0)
            # Update Q-values separately for each objective
            for objective, reward in rewards.items():
                # updated_Q = Q_current[objective] + self.learning_rate * (reward + self.discount_factor * Q_future[objective] - Q_current[objective])
                # self.Q[state_key][action][objective] = updated_Q
                # logging.info('Faraz - debug BEFORE update Q and reward: {}, {}'.format(self.Q[state_key][action][objective], reward))
                # Q_current[objective]+= self.learning_rate * (reward + self.discount_factor * Q_future[objective] - Q_current[objective])
                if round and objective=='accuracy':
                    Q_current[objective]+= learning_rate * reward
                else:
                    Q_current[objective]+= self.learning_rate * reward
                # logging.info('Faraz - debug AFTER update Q and reward: {}, {}'.format(self.Q[state_key][action][objective], reward))

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
            # Get the root logger
            root_logger = logging.getLogger()

            # Get the handlers of the root logger
            handlers = root_logger.handlers

            # Find the FileHandler (if it exists)
            file_handler = None
            for handler in handlers:
                if isinstance(handler, logging.FileHandler):
                    file_handler = handler
                    break

            # If a FileHandler is found, get the log file path
            # if file_handler:
            #     log_file_path = file_handler.baseFilename
            #     print("Log file path:", log_file_path)
            #     #remove log from path '/home/ahmad/FedScale/benchmark/logs/femnist/0728_070630/aggregator/log/Q.pkl'
            #     log_file_path = log_file_path.split('log')[0]
            path = os.path.join(path, 'Q.pkl')
            logging.info("Saving Q to {}".format(path))
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



