# -*- coding: utf-8 -*-
import copy
from collections import deque
from heapq import heappush, heappop
import numpy as np
from overrides import overrides
import math

from fedscale.cloud.fllibs import *
from fedscale.cloud.aggregation.aggregator import Aggregator


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
            if participation_feasibility:
                heappush(self.min_pq, (event_time, 'start', client))
                heappush(self.min_pq, (end_time, 'end', client))
                self.client_task_start_times[client] = event_time
                self.client_task_model_version[client] = self.round
            else:
                if len(self.clients_dropped_per_duration) > 0 and client in self.clients_dropped_per_duration[event_time]:
                    if self.event_type == 'start':
                        self.current_concurrency -= 1
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
            train_config = {'client_id': next_client_id, 'task_config': config}
            model_version = self.client_task_model_version[next_client_id]
            if len(self.model_cache) <= (self.round - model_version):
                return train_config, self.model_cache[-1]
        except Exception as e:
            logging.error("Faraz - Exception in create_client_task: {}".format(e))
            raise e
        return train_config, self.model_cache[self.round - model_version]

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
                    clients_before = len(clients_to_run)
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
                        durations.append(event_time - self.client_task_start_times[client])
                        final_time = event_time
                    # logging.info("HERE12")
                    clients_to_run = [client for client in clients_to_run if client not in self.clients_dropped_out_per_round]
                    logging.info("Clients to run: {}, num_clients_to_collect: {}, clients_dropped_out_per_round: {}".format(len(clients_to_run), num_clients_to_collect, self.clients_dropped_out_per_round))
                    # if clients_before == len(clients_to_run) and self.round >=199 and event_type != 'start':
                        # logging.info("Faraz - Increasing time because no clients were added to clients_to_run")
                        # event_time+=10
                        
                self.global_virtual_clock = max(final_time,self.global_virtual_clock)
                # logging.info("HERE13")
                
                #Faraz - remove dropped out clients in this round from the list of clients to run
                return clients_to_run, self.clients_dropped_out_per_round[self.round], self.virtual_client_clock, 0, durations
            else:
                # Dummy placeholder for non-simulations.
                completed_client_clock = {
                    client: {'computation': 1, 'communication': 1} for client in sampled_clients}
                times = [1 for _ in sampled_clients]
                return sampled_clients, sampled_clients, completed_client_clock, 1, times
        except Exception as e:
            logging.error("Faraz - Exception in tictak_client_tasks: {}".format(e))
            raise e

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
                logging.info("HERE35")
                update_weights = results['update_weight']
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
            
        
if __name__ == "__main__":
    aggregator = AsyncAggregator(parser.args)
    aggregator.run()
