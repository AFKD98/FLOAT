# -*- coding: utf-8 -*-
import copy
from collections import deque
from heapq import heappush, heappop
import numpy as np
from overrides import overrides
import math
from collections import deque

from fedscale.cloud.fllibs import *
from fedscale.cloud.aggregation.aggregator import Aggregator
from fedscale.utils.compressors.quantization import QSGDCompressor
from fedscale.utils.compressors.pruning import Pruning


class AsyncAggregator(Aggregator):
    """Represents an async aggregator implementing the FedBuff algorithm.
    Currently, this aggregator only supports simulation mode."""

    def _new_task(self, event_time):
        """Generates a new task that starts at event_time, and inserts it into the min heap.

        :param event_time: the time to start the new task.
        """
        try:
            if len(self.clients_dropped_per_duration) == 0:
                self.clients_dropped_per_duration[event_time] = []
            elif event_time > list(self.clients_dropped_per_duration.keys())[-1]:
                self.clients_dropped_per_duration = {}
                self.clients_dropped_per_duration[event_time] = []
            # logging.info("HERE40")
            client = self.client_manager.select_participants(1, cur_time=event_time)[0]
            # logging.info("Faraz - Selected client: {}".format(client))
            client_cfg = self.client_conf.get(client, self.args)
            # logging.info("HERE41")

            exe_cost = self.client_manager.get_completion_time_with_variable_network(
                client,
                batch_size=client_cfg.batch_size,
                local_steps=client_cfg.local_steps,
                upload_size=self.model_update_size,
                download_size=self.model_update_size)
            self.virtual_client_clock[client] = exe_cost
            duration = exe_cost['computation'] + \
                    exe_cost['communication']
            end_time = event_time + duration
            # Faraz - check if the client has enough resources to run the task
            # logging.info("HERE42")
            (participation_feasibility, deadline_difference) = self.client_manager.getClientFeasibilityForParticipation(client, event_time, self.model_update_size, self.model_update_size)
            # logging.info("HERE43")
            # logging.info(f'Faraz - Client {client} has {participation_feasibility} feasibility for participation')
            if participation_feasibility:
                heappush(self.min_pq, (event_time, 'start', client))
                heappush(self.min_pq, (end_time, 'end', client))
                self.client_task_start_times[client] = event_time
                self.client_task_model_version[client] = self.round
            else:
                client_active = False
                #Apply optimization to see if the client will be able to participate
                if not client_active:
                    action = self.get_optimization(client)
                    client_active, newRoundDuration, compressed_weights, new_exe_cost = self.perform_optimization(client_cfg, client, action, duration, exe_cost, event_time)
                    exe_cost = new_exe_cost
                    
                if client_active and new_exe_cost:
                    end_time = event_time + newRoundDuration
                    heappush(self.min_pq, (event_time, 'start', client))
                    heappush(self.min_pq, (end_time, 'end', client))
                    self.client_task_start_times[client] = event_time
                    self.client_task_model_version[client] = self.round
                    if self.optimizations.get(client) is None:
                        self.optimizations[client] = deque()
                    self.optimizations[client].append({'optimization': action, 'model_weights': compressed_weights if compressed_weights else self.model_wrapper.get_weights()})
                else:
                    if len(self.clients_dropped_per_duration) > 0 and client in self.clients_dropped_per_duration[event_time]:
                        if self.event_type == 'start':
                            self.current_concurrency -= 1
                        # logging.info("Faraz - Client {} dropped out".format(client))
                    else:
                        # print("Client {} dropped out".format(client))
                        client_resources = self.client_manager.get_client_resources(client)
                        self.update_client_dropped_out(client, exe_cost, deadline_difference, client_resources)
                        self.clients_dropped_per_duration[event_time].append(client)
                        heappush(self.min_pq, (event_time, 'start', client))
                        heappush(self.min_pq, (end_time, 'end', client))
                        self.client_task_start_times[client] = event_time
                        self.client_task_model_version[client] = self.round
            self.total_resources_selected_clients.append(exe_cost)
                # self.current_concurrency-=1
        except Exception as e:
            logging.error("Faraz - Exception in _new_task: {}".format(e))
            raise e


    @overrides
    def create_client_task(self, executor_id):
        """Issue a new client training task to the specific executor.

        Args:
            executor_id (int): Executor Id.

        Returns:
            tuple: Training config for new task. (dictionary, PyTorch or TensorFlow module)

        """
        try:
            next_client_id = self.resource_manager.get_next_task(executor_id)
            config = self.get_client_conf(next_client_id)
            if self.optimizations.get(next_client_id) is not None and len(self.optimizations[next_client_id])>0:
                action = self.optimizations[next_client_id].popleft()
                train_config = {'client_id': next_client_id, 'task_config': config, 'optimization': action['optimization']}
                # client_train_model = copy.deepcopy(action['model_weights'])
                client_train_model = copy.deepcopy(self.model_wrapper.get_weights())
                return train_config, client_train_model
            train_config = {'client_id': next_client_id, 'task_config': config, 'optimization': None}
            model_version = self.client_task_model_version[next_client_id]
            if len(self.model_cache) <= (self.round - model_version):
                return train_config, self.model_cache[-1]
        except Exception as e:
            logging.error("Faraz - Exception in create_client_task: {}".format(e))
            raise e
        return train_config, self.model_cache[self.round - model_version]


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
    def perform_optimization(self, client_cfg, client_to_run, optimization, oldroundDuration, exe_cost, event_time):
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
                

            isactive, olddeadline_difference = self.client_manager.isClientActivewithDeadline(client_to_run, oldroundDuration + event_time)
            client_active, deadline_difference = self.client_manager.isClientActivewithDeadline(client_to_run, roundDuration + event_time)
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
        # try:
        if self.experiment_mode == commons.SIMULATION_MODE:
            # logging.info("HERE10")
            self.model_cache.appendleft(self.model_wrapper.get_weights())
            if len(self.model_cache) > self.args.max_staleness + 1:
                self.model_cache.pop()
            clients_to_run = []
            durations = []
            final_time = self.global_virtual_clock
            if not self.min_pq:
                self.event_type = ''
                self._new_task(self.global_virtual_clock)
            # logging.info("HERE11")
            while len(clients_to_run) < num_clients_to_collect:
                #Faraz - clients to run before searching
                # logging.info('min_pq length: {}, clients_to_run length: {}'.format(len(self.min_pq), len(clients_to_run)))
                clients_before = len(clients_to_run)
                if len(self.min_pq) == 0:
                    self._new_task(final_time)
                event_time, event_type, client = heappop(self.min_pq)
                if event_type == 'start':
                    self.event_type = 'start'
                    self.current_concurrency += 1
                    if self.current_concurrency < self.args.max_concurrency:
                        self._new_task(event_time)
                else:
                    self.current_concurrency -= 1
                    if self.current_concurrency == self.args.max_concurrency - 1:
                        self.event_type = ''
                        self._new_task(event_time)
                    if self.round - self.client_task_model_version[client] <= self.args.max_staleness:
                        clients_to_run.append(client)
                    # else:
                        # logging.info("Faraz - Client {} model is stale by {} rounds".format(client, self.round - self.client_task_model_version[client]))
                    durations.append(event_time - self.client_task_start_times[client])
                    final_time = event_time
                # logging.info("HERE12")
                # logging.info("Clients to run BEFORE : {}, num_clients_to_collect: {}".format(len(clients_to_run), num_clients_to_collect))
                clients_to_run = [client for client in clients_to_run if client not in self.clients_dropped_out]
                # logging.info("Clients to run AFTER : {}, num_clients_to_collect: {}".format(len(clients_to_run), num_clients_to_collect))
                # if clients_before == len(clients_to_run) and self.round >=198:
                    # logging.info("Faraz - Increasing time because no clients were added to clients_to_run")
                    # event_time+=50
                    
            self.global_virtual_clock = max(final_time,self.global_virtual_clock)
            # logging.info("HERE13")
            
            #Faraz - remove dropped out clients in this round from the list of clients to run
            return clients_to_run, self.clients_dropped_out, self.virtual_client_clock, 0, durations
        else:
            # Dummy placeholder for non-simulations.
            completed_client_clock = {
                client: {'computation': 1, 'communication': 1} for client in sampled_clients}
            times = [1 for _ in sampled_clients]
            return sampled_clients, sampled_clients, completed_client_clock, 1, times
        # except Exception as e:
        #     logging.error("Faraz - Exception in tictak_client_tasks: {}".format(e))
        #     raise e

    @overrides
    def setup_env(self):
        """Set up environment and variables."""
        self.setup_seed(seed=1)
        self.virtual_client_clock = {}
        self.min_pq = []
        self.model_cache = deque()
        self.client_task_start_times = {}
        self.client_task_model_version = {}
        self.current_concurrency = 0
        self.aggregation_denominator = 0
        #Faraz - dictionary to keep track of dropped out clients per round
        self.clients_dropped_per_duration = {}
        self.event_type = ''
        self.q_compressor = QSGDCompressor(random=True, cuda=False)
        self.pruning = Pruning(cuda=True)
        self.train_dataset, self.test_dataset = init_dataset()
        logging.info('train_dataset type: {}'.format(type(self.train_dataset)))
        logging.info("Setting up environment")

    @overrides
    def update_weight_aggregation(self, results):
        """Updates the aggregation with the new results.

        Implements the aggregation mechanism implemented in FedBuff
        https://arxiv.org/pdf/2106.06639.pdf (Nguyen et al., 2022)

        :param results: the results collected from the client.
        """
        # logging.info("HERE35")
        
        try:
            if self.mode!='Fedbuff':
                # logging.info("HERE35")
                update_weights = results['update_weight']
                if results.get('optimization'):
                    n_bit = int(results['optimization'].split('_')[1])
                    update_weights = self.decompress_model(update_weights, n_bit)
                # Aggregation weight is derived from equation from "staleness scaling" section in the referenced FedBuff paper.
                inverted_staleness = 1 / (1 + self.round - self.client_task_model_version[results['client_id']]) ** 0.5
                self.aggregation_denominator += inverted_staleness
                if type(update_weights) is dict:
                    update_weights = [x for x in update_weights.values()]
                if self._is_first_result_in_round():
                    self.model_weights = [weight * inverted_staleness for weight in update_weights]
                else:
                    self.model_weights = [weight + inverted_staleness * update_weights[i] for i, weight in
                                        enumerate(self.model_weights)]
                if self._is_last_result_in_round():
                    self.model_weights = [np.divide(weight, self.aggregation_denominator) for weight in self.model_weights]
                    self.model_wrapper.set_weights(copy.deepcopy(self.model_weights))
                    self.aggregation_denominator = 0
            else:
                #Faraz - Implementing buffer aggregation for FedBuff
                #Faraz - buffer will be a list of dictionaries, each dictionary containing the update weights from a client
                client_weights = []
                # logging.info("HERE36")
                for _ in range(self.buffer_size):
                    # logging.info("buffer size: {}".format(len(self.client_results_buffer)))
                    client_data = self.client_results_buffer.popleft()
                    client_weights.append(client_data)
                # logging.info("HERE5")
                client_no = 0
                # logging.info("HERE37")
                # logging.info("Faraz BEFORE - self.model_weights weights: {}".format(self.model_weights))
                for client_weight in client_weights:
                    self.clients_aggregated.append(client_weight['client_id'])
                    if self.round not in self.clients_aggregated_per_round:
                        self.clients_aggregated_per_round[self.round] = []
                    else:
                        self.clients_aggregated_per_round[self.round].append(client_weight['client_id'])
                    update_weights = client_weight['update_weight']
                    if type(update_weights) is dict:
                        update_weights = [x for x in update_weights.values()]
                    # logging.info("HERE6")
                    # logging.info('update weights keys: {}'.format(update_weights.keys()))
                    # logging.info('update weights values: {}'.format(update_weights.values()))
                    # Aggregation weight is derived from equation from "staleness scaling" section in the referenced FedBuff paper.
                    inverted_staleness = 1 / (1 + self.round - self.client_task_model_version[client_weight['client_id']]) ** 0.5
                    self.aggregation_denominator += inverted_staleness
                    # logging.info("HERE7")
                    if client_no == 0:
                        self.model_weights = [weight * inverted_staleness for weight in update_weights]
                    else:
                        self.model_weights = [weight + inverted_staleness * update_weights[i] for i, weight in
                                            enumerate(self.model_weights)]
                    # logging.info("HERE8")
                    # logging.info("update weights shape: {}".format(self.model_weights))
                    client_no += 1
                # logging.info("HERE38")
                
                self.model_weights = [np.divide(weight, self.aggregation_denominator) for weight in self.model_weights]
                self.model_wrapper.set_weights(copy.deepcopy(self.model_weights))
                self.aggregation_denominator = 0
                # logging.info("Faraz AFTER - self.model_weights weights: {}".format(self.model_weights))
                
                # logging.info("HERE39")
                
        except Exception as e:
            logging.error("Error in update_weight_aggregation: {}".format(e))
            raise e
            

    @overrides
    def client_completion_handler(self, results):
        """We may need to keep all updates from clients,
        if so, we need to append results to the cache

        Args:
            results (dictionary): client's training result

        """
        try:
            # logging.info("HERE30")
            #Faraz- Implementing buffer handler for FedBuff
            # Format:
            #       -results = {'client_id':client_id, 'update_weight': model_param, 'moving_loss': round_train_loss,
            #       'trained_size': count, 'wall_duration': time_cost, 'success': is_success 'utility': utility}
            logging.info("Client {} completed round {} with utility {}".format(results['client_id'], self.round, results['utility']))
            if self.args.gradient_policy in ['q-fedavg']:
                self.client_training_results.append(results)
            # Feed metrics to client sampler
            # logging.info("HERE31")
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
            # Faraz - Check mode of aggregation
            # logging.info("HERE32")
            if self.mode == 'async':
                self.update_lock.acquire()

                self.model_in_update += 1
                self.update_weight_aggregation(results)

                self.update_lock.release()

            elif self.mode == 'Fedbuff':
                # logging.info("HERE33")
                self.client_results_buffer.append(results)
                # logging.info('length of client results buffer: {}'.format(len(self.client_results_buffer)))
                # logging.info('length of sampled participants: {}'.format(len(self.clients_to_run)))
                # logging.info('Buffer size: {}'.format(self.buffer_size))
                
                if len(self.client_results_buffer)>=self.buffer_size or len(self.client_results_buffer)==len(self.clients_to_run):
                    self.update_lock.acquire()
                    # logging.info("HERE34")
                    self.model_in_update += 1
                    self.update_weight_aggregation(results)
                    self.update_lock.release()
                    aggregation_performed = True
                else:
                    aggregation_performed = False
                    
                return aggregation_performed
        except Exception as e:
            logging.error("Error in client_completion_handler: {}".format(e))
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
        
if __name__ == "__main__":
    aggregator = AsyncAggregator(parser.args)
    aggregator.run()
