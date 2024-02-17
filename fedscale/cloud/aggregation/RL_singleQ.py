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
import math
# Set random seed for reproducibility
random.seed(0)
np.random.seed(0)


class RL:
    def __init__(self, total_clients=200, participation_rate=50, discount_factor=0, learning_rate=1.0, exploration_prob=0.6, actions=[1, 2]):
        self.total_clients = total_clients
        self.participation_rate = participation_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.actions = commons.ACTIONS
        self.w_p = 0.5
        self.w_a = 0.5
        self.rewards_per_round = []
        self.selected_actions_rewards = {}
        # path = '/home/ahmad/FedScale/benchmark/logs/rl_model/Q.pkl'
        Q_file = 'benchmark/logs/rl_model/Q.pkl'
        FLOAT_HOME = os.getcwd()
        FLOAT_HOME = os.path.join(FLOAT_HOME, 'FLOAT')
        self.filepath = os.path.join(FLOAT_HOME, Q_file)
        if os.path.exists(self.filepath):
            logging.info("Loading Q table from {}".format(self.filepath))
            self.get_Q_from_path(self.filepath)
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
                    logging.info("action q values: {}".format(action_rewards))
                    # Choose action based on the rewards for each objective
                    # action = max(objective_rewards, key=objective_rewards.get)
                    # Calculate the combined score for each entry
                    max_combined_score = 0.0
                    action = None

                    # Iterate through the data to find the key with the maximum combined score
                    average_acc = sum([value['accuracy'] for key, value in action_rewards.items()]) / len(action_rewards)
                    if average_acc < 0.5:
                        self.w_p = 0.5
                        self.w_a = 0.5
                    else:
                        self.w_p = 0.5
                        self.w_a = 0.5

                    for key, value in action_rewards.items():
                        combined_score = self.w_p*(value['participation_success']/value['count']) + (self.w_a*value['accuracy'])/value['count']
                        if combined_score > max_combined_score:
                            max_combined_score = combined_score
                            action = key
                    if action is None:
                        # action = np.random.choice(self.actions)
                        logging.info("State is still unexplored, Cannot exploit!")
                        action = self.choose_random_action_uniformly(state_key)
                    self.selected_actions_rewards[selected_client] = 0
                    # logging.info('choose - self.selected_actions_rewards: {}'.format(self.selected_actions_rewards))
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
            if selected_client in list(self.selected_actions_rewards.keys()):
                self.selected_actions_rewards[selected_client] += self.w_p*rewards['participation_success'] + (self.w_a*rewards['accuracy'][0] if isinstance(rewards['accuracy'], list) else rewards['accuracy'])
                participation_reward = self.w_p*rewards['participation_success']
                accuracy_reward = self.w_a*rewards['accuracy']
                calculated_reward = participation_reward + accuracy_reward
                self.rewards_per_round.append(calculated_reward)
                logging.info(f'Faraz - debug rewards: participation_success: {participation_reward}, accuracy: {accuracy_reward}, calculated_reward: {calculated_reward}')
                # logging.info(f'participation_success: {rewards["participation_success"]}, accuracy: {rewards["accuracy"]}')
                # logging.info('update - rewards: {}'.format(calculated_reward))
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
                # logging.info('BEFORE - objective: {}, value: {}'.format(objective, Q_current[objective]))
                if round and objective=='accuracy':
                    #check if Q_current[objective] == nan
                    Q_current[objective]+= learning_rate * reward
                    logging.info('AFTER - objective: {}, value: {}'.format(objective, Q_current[objective]))
                else:
                    Q_current[objective]+= self.learning_rate * reward
                # logging.info('AFTER - objective: {}, value: {}'.format(objective, Q_current[objective]))
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
            # return
            if os.path.exists(self.filepath):
                logging.info("Saving Q to {}".format(self.filepath))
                with open(self.filepath, 'wb') as f:
                    pickle.dump(self.Q, f)
            else:
                path = os.path.join(path, 'Q.pkl')
                logging.info("Saving Q to {}".format(path))
                with open(path, 'wb') as f:
                    pickle.dump(self.Q, f)
        except Exception as e:
            logging.error("Error in save_Q")
            logging.error(e)

    def get_Q_from_path(self, path):
        try:
            logging.info("Loading Q from {}".format(path))
            with open(path, 'rb') as f:
                self.Q = pickle.load(f)
        except Exception as e:
            logging.error("Error in get_Q_from_path")
            logging.error(e)
            
    def load_Q(self, path):
        try:
            if os.path.exists(self.filepath):
                logging.info("Loading Q to {}".format(self.filepath))
                with open(path, 'rb') as f:
                    self.Q = pickle.load(f)
            else:
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



