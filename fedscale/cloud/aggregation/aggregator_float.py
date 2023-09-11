# -*- coding: utf-8 -*-
import collections
import copy
import math
import os
import pickle
import random
import threading
import time
from concurrent import futures
import gc
import grpc
import numpy as np
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

import fedscale.cloud.channels.job_api_pb2_grpc as job_api_pb2_grpc
import fedscale.cloud.logger.aggregator_logging as logger
from fedscale.cloud.aggregation.optimizers import TorchServerOptimizer
from fedscale.cloud.channels import job_api_pb2
from fedscale.cloud.client_manager import ClientManager
from fedscale.cloud.internal.tensorflow_model_adapter import TensorflowModelAdapter
from fedscale.cloud.internal.torch_model_adapter import TorchModelAdapter
from fedscale.cloud.resource_manager import ResourceManager
from fedscale.cloud.fllibs import *
from torch.utils.tensorboard import SummaryWriter
#for reading bandwidth traces
import pandas as pd
#Faraz - For clustering resources in clients
from sklearn.cluster import KMeans
#Faraz - for quantizations
from fedscale.cloud.aggregation.RL import RL
from fedscale.utils.compressors.quantization import QSGDCompressor
# import pympler
from fedscale.cloud.aggregation.aggregator import Aggregator
from overrides import overrides
from fedscale.utils.compressors.pruning import Pruning

MAX_MESSAGE_LENGTH = 1 * 1024 * 1024 * 1024  # 1GB


#Faraz - Added Error reporting
class FLOATAggregator(Aggregator):
    """This centralized aggregator collects training/testing feedbacks from executors

    Args:
        args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py

    """
    @overrides
    def setup_env(self):
        """Set up environment and variables."""
        try:
            self.q_compressor = QSGDCompressor(random=True, cuda=False)
            self.pruning = Pruning(cuda=True)
            self.train_dataset, self.test_dataset = init_dataset()
            logging.info('train_dataset type: {}'.format(type(self.train_dataset)))
        except Exception as e:
            logging.error('Error in setup_env: ', e)
            raise e
        
    @overrides
    def get_optimization(self, client_to_run):
        try:
            '''Get the optimization method for the current client.'''
            client_local_state = self.client_manager.get_client_local_state(client_to_run)
            optimization = self.rl_agent.choose_action_per_client(self.global_state, client_local_state, client_to_run)
            return optimization
        except Exception as e:
            logging.error(f'Error in get_optimization: {e}')
            raise e
    
    
    @overrides
    def compress_model(self, optimization):
        try:
            compressed_weights = []
            total_size = 0
            # logging.info('HERE41')
            for weight in self.model_wrapper.get_weights():
                compressed_weight, size = self.q_compressor.compress(weight, n_bit=optimization)
                # logging.info('HERE42')
                total_size += int(size)
                # logging.info('HERE42')
                compressed_weights.append(compressed_weight)
                # logging.info('HERE43')
                
            return compressed_weights, total_size
        except Exception as e:
            logging.error('Error in compressing model: ', e)
            raise e
    
    def prune_model(self, prune_percentage=0.45):
        try:
            pruned_model, reduction_ratio = self.pruning.prune_model(self.model_wrapper.get_model(), prune_percentage, self.train_dataset)
            return pruned_model, reduction_ratio
        
        except Exception as e:
            logging.error('Error in pruning model: ', e)
            raise e

    @overrides
    def decompress_model(self, model, optimization=8):
        try:
            decompressed_weights = []
            for weight in model:
                decompressed_weight = self.q_compressor.decompress(weight, n_bit=optimization)
                decompressed_weights.append(decompressed_weight)
            return decompressed_weights
        except Exception as e:
            logging.error('Error in decompressing model: ', e)
            raise e
    @overrides
    def update_RL_agent(self):
        '''Update the RL agent with the current client's information.'''
        try:

            #Faraz - sum of rewards
            # logging.info('sum of rewards: {}'.format(sum(self.rl_agent.selected_actions_rewards.values())))
            exploited_client_ids = list(self.rl_agent.selected_actions_rewards.keys())
            for client_id, update in self.past_rl_updates.items():
                if 'global_state' in update:
                    global_state = update['global_state']
                    local_state = update['local_state']
                    optimization = update['optimization']
                    new_global_state = update['new_global_state']
                    new_local_state = update['new_local_state']
                    reward = update['reward']
                    reward['accuracy'] = np.mean(reward['accuracy'])
                    self.rl_agent.update_Q_per_client(client_id, global_state, local_state, optimization, new_global_state, new_local_state, reward, self.round)
                    self.rl_agent.save_Q('/home/ahmad/FedScale/benchmark/logs/rl_model')
                    #remove from rl_updates
                    if client_id in exploited_client_ids:
                        self.rl_agent.selected_actions_rewards.pop(client_id)
                    # logging.info(f'Updated RL Q table: {self.rl_agent.Q}')
                else:
                    logging.info('No update for RL agent')
            # logging.info('update_RL_agent: rl_updates: {}'.format(self.rl_updates))
            logging.info('Faraz - Rewards in round {}: {}'.format(self.round, self.rl_agent.rewards_per_round))
            logging.info(f'Sum of rewards in round {self.round}: {sum(self.rl_agent.rewards_per_round)}')
            
            # self.rl_agent.print_overhead_times()
        except Exception as e:
            logging.error('Error in updating RL agent: ', e)
            raise e

    @overrides
    def perform_optimization(self, client_cfg, client_to_run, optimization, oldroundDuration, exe_cost):
        '''Perform the optimization method for the current client.'''
        try:
            client_local_state = self.client_manager.get_client_local_state(client_to_run)
            # logging.info('HERE40')
            compressed_weights = None
            logging.info(f'Faraz - optimization: {optimization} for client {client_to_run}')
            if 'quantization' in optimization:
                q_bit = int(optimization.split('_')[1])
                compressed_weights, size = self.compress_model(q_bit)
                logging.info(f"Faraz - Compressed model size: {size / 1024.0 * 8}, seize before compression: {self.model_update_size}")
                size =  size / 1024.0 * 8.  # kbits
                exe_cost = self.client_manager.get_completion_time_with_variable_network(
                            client_to_run,
                            batch_size=client_cfg.batch_size,
                            local_steps=client_cfg.local_steps,
                            upload_size=size,
                            download_size=self.model_update_size)
                
                roundDuration = exe_cost['computation'] + \
                                            exe_cost['communication']
            elif 'pruning' in optimization:
                prune_percentage = int(optimization.split('_')[1])*0.01
                # pruned_model, reduction_ratio = self.prune_model(prune_percentage)
                #Faraz - debug - temporarily using prune percentage as reduction ratio
                reduction_ratio = 1.0-prune_percentage
                roundDuration = exe_cost['computation']*reduction_ratio + exe_cost['communication']*reduction_ratio
            elif 'partial' in optimization:
                partial_training_percentage = 1.0 - int(optimization.split('_')[1])*0.01
                # pruned_model, reduction_ratio = self.prune_model(partial_training_percentage)
                #Faraz - debug - temporarily using prune percentage as reduction ratio
                new_local_steps = int((partial_training_percentage)*self.args.local_steps)
                # logging.info(f'Faraz - debug old vs new local steps and partial_training_percentage: {self.args.local_steps}, {new_local_steps}, {partial_training_percentage}')
                roundDuration = (exe_cost['computation']//self.args.local_steps)*new_local_steps + exe_cost['communication']
                

            isactive, olddeadline_difference = self.client_manager.isClientActivewithDeadline(client_to_run, oldroundDuration + self.global_virtual_clock)
            client_active, deadline_difference = self.client_manager.isClientActivewithDeadline(client_to_run, roundDuration + self.global_virtual_clock)
            logging.info(f"Faraz - Client {client_to_run} is active: {client_active}, deadline difference: {deadline_difference}, old deadline difference: {olddeadline_difference}, round duration difference: {abs(oldroundDuration - roundDuration)}")
            if client_active:
                logging.info('Faraz - Successfully scheduled client {} for round {} with optimization {} and round duration reduction of {}%'.format(client_to_run, self.round, optimization, oldroundDuration - roundDuration))
                
                # self.rl_agent.update_Q_per_client(client_to_run, self.global_state, client_local_state, optimization, self.global_state, client_local_state, 1)
                self.rl_updates[client_to_run] = {'client_to_run': client_to_run, 'global_state': self.global_state, 'local_state': client_local_state, 'optimization': optimization, 'new_global_state': self.global_state, 'new_local_state': client_local_state, 'reward': {'participation_success': 1.0, 'accuracy': []}}
                logging.info('rl_updates: {}'.format(self.rl_updates))
                self.rl_agent.print_overhead_times()
                if 'quantization' in optimization:
                    return True, roundDuration, compressed_weights, exe_cost
                elif 'pruning' in optimization:
                    return True, roundDuration, None, exe_cost
                elif 'partial' in optimization:
                    return True, roundDuration, None, exe_cost
                
            else:
                participationSuccess = -1.0
                if int(optimization.split('_')[1])>=75:
                    participationSuccess = 1.0
                logging.info('Faraz - Failed to schedule client {} for round {} with optimization {} and round duration reduction of {}%'.format(client_to_run, self.round, optimization, abs(oldroundDuration - roundDuration)))
                # self.rl_agent.update_Q_per_client(client_to_run, self.global_state, client_local_state, optimization, self.global_state, client_local_state, -1)
                self.rl_updates[client_to_run] =  {'client_to_run': client_to_run, 'global_state': self.global_state, 'local_state': client_local_state, 'optimization': optimization, 'new_global_state': self.global_state, 'new_local_state': client_local_state, 'reward': {'participation_success': participationSuccess, 'accuracy': []}}
                logging.info('rl_updates: {}'.format(self.rl_updates))
                self.rl_agent.print_overhead_times()
                if 'quantization' in optimization:
                    return False, roundDuration, compressed_weights, exe_cost
                elif 'pruning' in optimization:
                    return False, roundDuration, None, exe_cost
                elif 'partial' in optimization:
                    return False, roundDuration, None, exe_cost
        except Exception as e:
            logging.error('Error in performing optimization: ', e)
            raise e
            
        
    @overrides
    def tictak_client_tasks(self, sampled_clients, num_clients_to_collect):
        """Record sampled client execution information in last round. In the SIMULATION_MODE,
        further filter the sampled_client and pick the top num_clients_to_collect clients.

        Args:
            sampled_clients (list of int): Sampled clients from client manager
            num_clients_to_collect (int): The number of clients actually needed for next round.

        Returns:
            Tuple: (the List of clients to run, the List of stragglers in the round, a Dict of the virtual clock of each
            client, the duration of the aggregation round, and the durations of each client's task).

        """
        try:
            if self.experiment_mode == commons.SIMULATION_MODE:
                # NOTE: We try to remove dummy events as much as possible in simulations,
                # by removing the stragglers/offline clients in overcommitment"""
                sampledClientsReal = []
                completionTimes = []
                completed_client_clock = {}
                client_completion_times = {}
                client_resources = {}
                # 1. remove dummy clients that are not available to the end of training
                clients_left_out = []
                for client_to_run in sampled_clients:
                    client_cfg = self.client_conf.get(client_to_run, self.args)

                    # exe_cost = self.client_manager.get_completion_time(client_to_run,
                    #                                                 batch_size=client_cfg.batch_size,
                    #                                                 local_steps=client_cfg.local_steps,
                    #                                                 upload_size=self.model_update_size,
                    #                                                 download_size=self.model_update_size)
                    exe_cost = self.client_manager.get_completion_time_with_variable_network(
                    client_to_run,
                    batch_size=client_cfg.batch_size,
                    local_steps=client_cfg.local_steps,
                    upload_size=self.model_update_size,
                    download_size=self
                    .model_update_size)

                    roundDuration = exe_cost['computation'] + \
                                    exe_cost['communication']
                    client_active, deadline_difference = self.client_manager.isClientActivewithDeadline(client_to_run, roundDuration + self.global_virtual_clock)
                    newRoundDuration, compressed_weights, new_exe_cost = roundDuration, None, None
                    action = None
                    if not client_active:
                        action = self.get_optimization(client_to_run)
                        #Faraz - for choosing static action
                        #Faraz - for testing individual optimization
                        # action = 'partial_50'
                        client_active, newRoundDuration, compressed_weights, new_exe_cost = self.perform_optimization(client_cfg, client_to_run, action, roundDuration, exe_cost)
                    # logging.info('tictak clients: self.rl_updates: {}'.format(self.rl_updates))
                    # if the client is not active by the time of collection, we consider it is lost in this round
                    
                    if client_active and new_exe_cost:
                        sampledClientsReal.append(client_to_run)
                        completionTimes.append(newRoundDuration)
                        self.optimizations[client_to_run] = {'optimization': action, 'model_weights': compressed_weights if compressed_weights else self.model_wrapper.get_weights()}
                        completed_client_clock[client_to_run] = new_exe_cost
                        client_completion_times[client_to_run] = newRoundDuration
                        client_resources[client_to_run] = new_exe_cost
                    elif client_active:
                        sampledClientsReal.append(client_to_run)
                        completionTimes.append(roundDuration)
                        completed_client_clock[client_to_run] = exe_cost
                        client_completion_times[client_to_run] = roundDuration
                        client_resources[client_to_run] = exe_cost
                    else:
                        clients_left_out.append(client_to_run)
                        sampledClientsReal.append(client_to_run)
                        self.optimizations[client_to_run] = {'optimization': action, 'model_weights': compressed_weights if compressed_weights else self.model_wrapper.get_weights()}
                        completionTimes.append(roundDuration)
                        completed_client_clock[client_to_run] = exe_cost
                        client_completion_times[client_to_run] = roundDuration
                        client_resources[client_to_run] = exe_cost
                    # else:
                    #     self.update_client_dropped_out(client_to_run, exe_cost, deadline_difference, client_resources)

                num_clients_to_collect = min(
                    num_clients_to_collect, len(completionTimes))
                # 2. get the top-k completions to remove stragglers
                workers_sorted_by_completion_time = sorted(
                    range(len(completionTimes)), key=lambda k: completionTimes[k])
                # top_k_index = workers_sorted_by_completion_time[:num_clients_to_collect]
                # clients_to_run = [sampledClientsReal[k] for k in top_k_index]
                clients_to_run = sampledClientsReal
                
                clients_to_run = random.sample(clients_to_run, num_clients_to_collect)
                    
                #Faraz - remove clients dropped out
                real_clients_to_run = [client for client in clients_to_run if client not in clients_left_out]
                #Faraz - find clients in clients_to_run but not in real_clients_to_run
                clients_dropped_out = [client for client in clients_to_run if client not in real_clients_to_run]
                self.clients_dropped_out = clients_dropped_out
                self.clients_dropped_out_per_round[self.round] = clients_dropped_out
                
                for client in clients_dropped_out:
                    self.dropped_clients_resource_usage.append(client_resources[client])
                    self.dropped_clients_resource_durations_per_round[self.round].append(client_resources[client])
                    self.dropped_clients_resource_durations_per_round[self.round].append(client_resources[client])
                
                #Faraz - collect resource usage and completion time for selected clients
                for client in clients_to_run:
                    self.total_resources_selected_clients.append(client_resources[client])
                    
                #Faraz - remove dropped clients from clients to run
                                #Faraz - remove dropped clients from clients to run
                #Faraz -  temporarily turned off to  get all clients training accuracy
                clients_to_run = real_clients_to_run
                 
                stragglers = [sampledClientsReal[k]
                            for k in workers_sorted_by_completion_time[num_clients_to_collect:]]
                
                max_time = 0
                if len(real_clients_to_run) > 0:
                    for client in real_clients_to_run:
                        if client_completion_times[client] > max_time:
                            max_time = client_completion_times[client]
                    round_duration = max_time
                else:
                    round_duration = max_time
                
                completionTimes.sort()
                
                return (clients_to_run, stragglers,
                        completed_client_clock, round_duration,
                        completionTimes[:num_clients_to_collect])
            else:
                completed_client_clock = {
                    client: {'computation': 1, 'communication': 1} for client in sampled_clients}
                completionTimes = [1 for c in sampled_clients]
                return (sampled_clients, sampled_clients, completed_client_clock,
                        1, completionTimes)
        except Exception as e:
            logging.error('Error in tictak_client_tasks: ', e)
            raise e

    @overrides
    def run(self):
        """Start running the aggregator server by setting up execution
        and communication environment, and monitoring the grpc message.
        """
        try:
            self.setup_env()
            self.client_profiles = self.load_client_profile(
                file_path=self.args.device_conf_file)
                
            self.init_control_communication()
            self.init_data_communication()

            self.init_model()
            self.model_update_size = sys.getsizeof(
                pickle.dumps(self.model_wrapper)) / 1024.0 * 8.  # kbits

            self.event_monitor()
            self.stop()
        except Exception as e:
            logging.error('Error in running aggregator: ', e)
            raise e


    @overrides
    def update_weight_aggregation(self, results):
        """Updates the aggregation with the new results.

        :param results: the results collected from the client.
        """
        try:
            update_weights = results['update_weight']
            if results.get('optimization'):
                # logging.info(f'type of update_weights: {type(update_weights)}')
                # logging.info(f'update_weights: {(update_weights)}')
                n_bit = int(results['optimization'].split('_')[1])
                update_weights = self.decompress_model(update_weights, n_bit)
                # logging.info(f'decompressed update_weights: {(update_weights)}')
            if type(update_weights) is dict:
                update_weights = [x for x in update_weights.values()]
            if type(update_weights[0]) != np.ndarray:
                update_weights = [np.array(x) for x in update_weights]
            if self._is_first_result_in_round():
                self.model_weights = update_weights
            else:
                # logging.info(f'Faraz - Starting aggregating, update weights type: {type(update_weights[0])}')
                
                self.model_weights = [weight + update_weights[i] for i, weight in enumerate(self.model_weights)]
                # logging.info(f'Faraz - Finished aggregating')
            if self._is_last_result_in_round():
                self.model_weights = [np.divide(weight, self.tasks_round) for weight in self.model_weights]
                self.model_wrapper.set_weights(copy.deepcopy(self.model_weights))
        except Exception as e:
            logging.error('Error in update weight aggregation: ', e)
            raise e


    @overrides
    def round_completion_handler(self):
        """Triggered upon the round completion, it registers the last round execution info,
        broadcast new tasks for executors and select clients for next round.
        """
        try:
            # logging.info("HERE3")
            self.global_virtual_clock += self.round_duration
            self.round += 1
            # logging.info('round_completion_handler1: rl_updates: {}'.format(self.rl_updates))

            last_round_avg_util = sum(self.stats_util_accumulator) / max(1, len(self.stats_util_accumulator))
            # assign avg reward to explored, but not ran workers
            for client_id in self.round_stragglers:
                self.client_manager.register_feedback(client_id, last_round_avg_util,
                                                    time_stamp=self.round,
                                                    duration=self.virtual_client_clock[client_id]['computation'] +
                                                            self.virtual_client_clock[client_id]['communication'],
                                                    success=False)
            # logging.info("HERE4")

            avg_loss = sum(self.loss_accumulator) / max(1, len(self.loss_accumulator))
            logging.info(f"Wall clock: {round(self.global_virtual_clock)} s, round: {self.round}, Planned participants: " +
                        f"{len(self.sampled_participants)}, Succeed participants: {len(self.stats_util_accumulator)}, Training loss: {avg_loss}")
            # logging.info("HERE5")

            # dump round completion information to tensorboard
            if len(self.loss_accumulator):
                self.log_train_result(avg_loss)
            # logging.info("HERE6")
            logging.info('Faraz - getting clients at time: {}'.format(self.global_virtual_clock))
            
            # update select participants
            self.sampled_participants = self.select_participants(
                select_num_participants=self.args.num_participants, overcommitment=self.args.overcommitment)
            # logging.info("HERE7")
            #Faraz - update client participation history
            for client_id in self.sampled_participants:
                if client_id not in self.client_participation_rate:
                    self.client_participation_rate[client_id] = []
                else:
                    self.client_participation_rate[client_id] +=1

            #Faraz - incase no clients dropped out
            if self.round not in self.clients_dropped_out_per_round:
                self.clients_dropped_out_per_round[self.round] = []
                self.dropped_clients_resource_usage_per_round[self.round] = []
                self.dropped_clients_resource_durations_per_round[self.round] = []
                self.deadline_differences_per_round[self.round] = []
            
            if self.rl_updates != {}:
                self.past_rl_updates = copy.deepcopy(self.rl_updates)
                #Faraz- Reset the updates
                self.rl_updates = {}
                gc.collect()

            (clients_to_run, round_stragglers, virtual_client_clock, round_duration,
            flatten_client_duration) = self.tictak_client_tasks(
                self.sampled_participants, self.args.num_participants)
            self.clients_to_run = clients_to_run
            #Faraz - update participation per round
            self.participation_per_round[self.round] = len(self.sampled_participants) - len(self.clients_dropped_out_per_round[self.round])
            
            # logging.info("HERE8")
            self.print_stats(clients_to_run)
            # logging.info("HERE9")
            # logging.info('round_completion_handler2: self.rl_updates: {}'.format(self.rl_updates))
            if len(self.clients_to_run) > 0:
                # Issue requests to the resource manager; Tasks ordered by the completion time
                self.resource_manager.register_tasks(clients_to_run)
                self.tasks_round = len(clients_to_run)

                # Update executors and participants
                if self.experiment_mode == commons.SIMULATION_MODE:
                    self.sampled_executors = list(
                        self.individual_client_events.keys())
                else:
                    self.sampled_executors = [str(c_id)
                                            for c_id in self.sampled_participants]
                self.round_stragglers = round_stragglers
                self.virtual_client_clock = virtual_client_clock
                self.flatten_client_duration = np.array(flatten_client_duration)
                self.round_duration = round_duration
                self.model_in_update = 0
                self.test_result_accumulator = []
                self.stats_util_accumulator = []
                self.client_training_results = []
                self.loss_accumulator = []
                self.update_default_task_config()

                if self.round >= self.args.rounds:
                    self.broadcast_aggregator_events(commons.SHUT_DOWN)
                # if self.round % self.args.eval_interval == 0 or self.round == 1:
                # if self.round % 50 == 0 or self.round == 1:
                #Faraz - validate missed(succeeded + dropped) clients every round
                # logging.info('before going in rl_updates: {}'.format(self.rl_updates))
                # if self.rl_updates!={} and self.round!= 1:
                #     logging.info('HERE506')
                #     # logging.info('Going in rl_updates: {}'.format(self.rl_updates))
                #     self.broadcast_aggregator_events(commons.UPDATE_MODEL)
                #     self.broadcast_aggregator_events(commons.CLIENT_VALIDATE)
                    # self.update_RL_agent()
                    # #Faraz- Reset the updates
                    # self.rl_updates = {}
                    # gc.collect()
                
                # if self.rl_updates != {}:
                #     self.update_RL_agent()
                #     self.broadcast_aggregator_events(commons.UPDATE_MODEL)
                #     self.broadcast_aggregator_events(commons.CLIENT_VALIDATE)
                if self.round % 50 == 0:
                    self.broadcast_aggregator_events(commons.UPDATE_MODEL)
                    self.broadcast_aggregator_events(commons.MODEL_TEST)
                    # self.rl_agent.save_Q('rl_agent')
                    if self.round % 50 == 0:
                        logging.info('Faraz - debug Sending validate all request')
                        self.broadcast_aggregator_events(commons.CLIENT_VALIDATE_ALL)
                else:
                    
                    self.broadcast_aggregator_events(commons.UPDATE_MODEL)
                    self.broadcast_aggregator_events(commons.START_ROUND)
                    # logging.info('HERE503')
                # if self.rl_updates != {}:
                #     self.broadcast_aggregator_events(commons.UPDATE_MODEL)
                #     self.broadcast_aggregator_events(commons.CLIENT_VALIDATE)
                #     logging.info(f'Faraz - len(self.stats_util_accumulator): {len(self.stats_util_accumulator)}')
                
                
            else:
                #Faraz - Skip round if no clients to run
                self.round_completion_handler()
                logging.info('No clients to run, skipping round')
        except Exception as e:
            logging.error('Error in round completion handler: ', e)
            raise e

    @overrides
    def validation_completion_handler(self, client_id, results):
        """Each executor will handle a subset of validation dataset

        Args:
            client_id (int): The client id.
            results (dictionary): The client validation accuracies.

        """
        try:
            logging.info('Validation completion handler results: {}'.format(results))
            # logging.info('validation_completion_handler: BEFORE self.rl_updates: {}'.format(self.rl_updates))
            for client_id, accuracy in results.items():
                # logging.info('Client {} validation accuracy: {}'.format(client_id, accuracy))
                if client_id not in self.past_rl_updates:
                    self.past_rl_updates[client_id] = {}
                    if 'reward' not in self.past_rl_updates[client_id]:
                        self.past_rl_updates[client_id]['reward'] = {}
                        self.past_rl_updates[client_id]['reward']['accuracy'] = []
                # if accuracy > 0:
                #     self.past_rl_updates[client_id]['reward']['accuracy'].append(1)
                # else:
                #     self.past_rl_updates[client_id]['reward']['accuracy'].append(-1)
                self.past_rl_updates[client_id]['reward']['accuracy'].append(accuracy)
            #Faraz - update RL agent after every client validation
            logging.info('past_rl_updates: {} in round: {}'.format(self.past_rl_updates, self.round))
            self.update_RL_agent()
            self.past_rl_updates = {}
            
            # logging.info('validation_completion_handler: AFTER self.rl_updates: {}'.format(self.rl_updates))

        except Exception as e:
            logging.error('Error in validation completion handler: ', e)
            raise e

    @overrides
    def client_completion_handler(self, results):
        """We may need to keep all updates from clients,
        if so, we need to append results to the cache

        Args:
            results (dictionary): client's training result

        """
        # Format:
        #       -results = {'client_id':client_id, 'update_weight': model_param, 'moving_loss': round_train_loss,
        #       'trained_size': count, 'wall_duration': time_cost, 'success': is_success 'utility': utility}
        try:
            client_id = results['client_id']
            # logging.info(f'Client {client_id} completed training.')
            if self.args.gradient_policy in ['q-fedavg']:
                self.client_training_results.append(results)
            # Feed metrics to client sampler
            self.stats_util_accumulator.append(results['utility'])
            self.loss_accumulator.append(results['moving_loss'])

            self.client_manager.register_feedback(results['client_id'], results['utility'],
                                                auxi=math.sqrt(
                                                    results['moving_loss']),
                                                time_stamp=self.round,
                                                duration=self.virtual_client_clock[results['client_id']]['computation'] +
                                                        self.virtual_client_clock[results['client_id']]['communication']
                                                )

            # ================== Aggregate weights ======================
            self.update_lock.acquire()
            if client_id not in self.past_clients_dropped_out:
                # logging.info(f'Faraz - debug Client {client_id} not dropped out, aggregate weights')
                self.model_in_update += 1
                self.update_weight_aggregation(results)
            else:
                logging.info(f'Faraz - debug Client {client_id} dropped out, skip aggregation')

            self.update_lock.release()
        except Exception as e:
            logging.error('Error in client completion handler: ', e)
            raise e
        
    @overrides
    def broadcast_aggregator_events(self, event):
        """Issue tasks (events) to aggregator worker processes by adding grpc request event
        (e.g. MODEL_TEST, MODEL_TRAIN) to event_queue.

        Args:
            event (string): grpc event (e.g. MODEL_TEST, MODEL_TRAIN) to event_queue.

        """
        self.broadcast_events_queue.append(event)

    @overrides
    def create_client_task(self, executor_id):
        """Issue a new client training task to specific executor

        Args:
            executorId (int): Executor Id.

        Returns:
            tuple: Training config for new task. (dictionary, PyTorch or TensorFlow module)

        """
        #Faraz - To do: add information about optimiztaion here or the model
        # temp_weights = self.model_wrapper.get_weights()
        # logging.info(f'np.srray(temp_weights): {np.array(temp_weights).shape})')
        # for w in temp_weights:
        #     logging.info('model weights: {}'.format(w.shape))
        try:
            # logging.info('executor_id: {}'.format(executor_id))
            next_client_id = self.resource_manager.get_next_task(executor_id)
            train_config = None
            # NOTE: model = None then the executor will load the global model broadcasted in UPDATE_MODEL
            if next_client_id is not None:
                    
                config = self.get_client_conf(next_client_id)
            if self.optimizations.get(next_client_id) is not None:
                train_config = {'client_id': next_client_id, 'task_config': config, 'optimization': self.optimizations[next_client_id]['optimization']}
                client_train_model = copy.deepcopy(self.optimizations[next_client_id]['model_weights'])
                # logging.info('HERE44')
                self.optimizations[next_client_id] = None
                return train_config, client_train_model
            # logging.info('HERE45')
            # logging.info('next_client_id: {}'.format(next_client_id))
            train_config = {'client_id': next_client_id, 'task_config': config, 'optimization': None}
            return train_config, self.model_wrapper.get_weights()
        except Exception as e:
            logging.error('Error in create client task: ', e)
            raise e

    @overrides
    def get_val_config(self, clients=None):
        """FL model testing on clients, developers can further define personalized client config here.

        Args:
            client_id (int): The client id.

        Returns:
            dictionary: The testing config for new task.

        """
        # logging.info('get_val_config self.rl_updates: {}'.format(self.rl_updates))
        try:
            if clients:
                return {'round': self.round, 'clients': clients}
            return {'round': self.round, 'clients': self.client_manager.getAllClients()}
        except Exception as e:
            logging.error(f'Error in get_val_config: {e}')
            raise e
    

    @overrides
    def CLIENT_EXECUTE_COMPLETION(self, request, context):
        """FL clients complete the execution task.

        Args:
            request (CompleteRequest): Complete request info from executor.

        Returns:
            ServerResponse: Server response to job completion request

        """
        try:
            executor_id, client_id, event = request.executor_id, request.client_id, request.event
            execution_status, execution_msg = request.status, request.msg
            meta_result, data_result = request.meta_result, request.data_result

            if event == commons.CLIENT_TRAIN:
                # Training results may be uploaded in CLIENT_EXECUTE_RESULT request later,
                # so we need to specify whether to ask client to do so (in case of straggler/timeout in real FL).
                if execution_status is False:
                    logging.error(f"Executor {executor_id} fails to run client {client_id}, due to {execution_msg}")

                # TODO: whether we should schedule tasks when client_ping or client_complete
                if self.resource_manager.has_next_task(executor_id):
                    # NOTE: we do not pop the train immediately in simulation mode,
                    # since the executor may run multiple clients
                    if commons.CLIENT_TRAIN not in self.individual_client_events[executor_id]:
                        # logging.info(f"Faraz - debug Executor {executor_id} has next task, schedule it")
                        #Faraz - To do insert the validation here
                        self.individual_client_events[executor_id].append(
                            commons.CLIENT_TRAIN)
                elif not self.resource_manager.has_next_task(executor_id) and self.rl_updates != {}:
                    #Faraz - Add update and validate tasks
                    logging.info(f"Faraz - debug Executor {executor_id} has no next task, schedule update and validate")
                    self.individual_client_events[executor_id].append(commons.UPDATE_MODEL)
                    self.individual_client_events[executor_id].append(commons.CLIENT_VALIDATE)
                

            elif event in (commons.MODEL_TEST, commons.UPLOAD_MODEL, commons.CLIENT_VALIDATE, commons.CLIENT_VALIDATE_ALL):
                self.add_event_handler(
                    executor_id, event, meta_result, data_result)
            else:
                logging.error(f"Received undefined event {event} from client {client_id}")

            return self.CLIENT_PING(request, context)
        except Exception as e:
            logging.error(f"Error in CLIENT_EXECUTE_COMPLETION: {e}")
            raise e

    @overrides
    def event_monitor(self):
        """Activate event handler according to the received new message
        """
        logging.info("Start monitoring events ...")
        try:
            while True:
                # Broadcast events to clients
                if len(self.broadcast_events_queue) > 0:
                    try:
                        current_event = self.broadcast_events_queue.popleft()
                        # logging.info("HERE20")
                        if current_event in (commons.UPDATE_MODEL, commons.MODEL_TEST, commons.CLIENT_VALIDATE, commons.CLIENT_VALIDATE_ALL):
                            self.dispatch_client_events(current_event)
                            # logging.info("HERE21")
                        elif current_event == commons.START_ROUND:

                            self.dispatch_client_events(commons.CLIENT_TRAIN)
                            # logging.info("HERE22")
                        elif current_event == commons.SHUT_DOWN:
                            self.dispatch_client_events(commons.SHUT_DOWN)
                            break
                        # logging.info("HERE23")
                        
                    except Exception as e:
                        logging.error(f"Error in event monitor broadcast events to clients: {e}")

                # Handle events queued on the aggregator
                elif len(self.sever_events_queue) > 0:
                    try:
                        client_id, current_event, meta, data = self.sever_events_queue.popleft()

                        if current_event == commons.UPLOAD_MODEL:
                            aggregation_performed = self.client_completion_handler(
                                self.deserialize_response(data))
                            # logging.info("HERE1")
                            if self.mode == 'async':
                                logging.info(f"Faraz - Async mode: {len(self.stats_util_accumulator)}")
                                if len(self.stats_util_accumulator) == self.tasks_round:
                                    self.round_completion_handler()
                            elif self.mode == 'Fedbuff':
                                #TODO: Call round_completion_handler() if buffer is full
                                # if len(self.stats_util_accumulator) == self.buffer_size or aggregation_performed:
                                if len(self.stats_util_accumulator) == self.buffer_size or aggregation_performed:
                                    self.round_completion_handler()
                                # logging.info("HERE2")
                            elif self.mode == 'oort':
                                if len(self.stats_util_accumulator) == self.tasks_round:
                                    self.round_completion_handler()
                            elif self.mode == 'FLOAT':
                                if len(self.stats_util_accumulator) == self.tasks_round:
                                    self.round_completion_handler()
                            else:
                                if len(self.stats_util_accumulator) == self.tasks_round:
                                    self.round_completion_handler()

                        elif current_event == commons.MODEL_TEST:
                            self.testing_completion_handler(
                                client_id, self.deserialize_response(data))
                            
                        elif current_event == commons.CLIENT_VALIDATE:
                            self.validation_completion_handler(
                                client_id, self.deserialize_response(data))
                        elif current_event == commons.CLIENT_VALIDATE_ALL:
                            continue

                        else:
                            logging.error(f"Event {current_event} is not defined")
                    except Exception as e:
                        logging.error(f"Error in event_monitor aggregator queue: {e}")

                else:
                    # execute every 100 ms
                    time.sleep(0.1)
                    # logging.info("HERE100")
        except Exception as e:
            logging.error(f"Error in event_monitor: {e}")


if __name__ == "__main__":
    try:
        aggregator = FLOATAggregator(parser.args)
        aggregator.run()
    except Exception as e:
        logging.error(f"Error in running aggregator: {e}")
        raise e
