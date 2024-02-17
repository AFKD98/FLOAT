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
# from fedscale.cloud.aggregation.RL import RL
from fedscale.cloud.aggregation.RL_singleQ import RL as RL_singleQ
from fedscale.utils.compressors.quantization import QSGDCompressor
# import pympler

MAX_MESSAGE_LENGTH = 1 * 1024 * 1024 * 1024  # 1GB


#Faraz - Added Error reporting
class Aggregator(job_api_pb2_grpc.JobServiceServicer):
    """This centralized aggregator collects training/testing feedbacks from executors

    Args:
        args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py

    """
    def __init__(self, args):
        # init aggregator loger
        # try:
        print("Aggregator init")
        logger.initiate_aggregator_setting()

        logging.info(f"Job args {args}")
        self.args = args
        self.experiment_mode = args.experiment_mode
        self.device = args.cuda_device if args.use_cuda else torch.device(
            'cpu')

        # ======== env information ========
        self.this_rank = 0
        self.global_virtual_clock = 0.
        self.round_duration = 0.
        self.resource_manager = ResourceManager(self.experiment_mode)
        self.client_manager = self.init_client_manager(args=args)

        # ======== model and data ========
        self.model_wrapper = None
        self.model_in_update = 0
        self.update_lock = threading.Lock()
        # all weights including bias/#_batch_tracked (e.g., state_dict)
        self.model_weights = None
        self.temp_model_path = os.path.join(
            logger.logDir, 'model_'+str(args.this_rank)+".npy")
        self.last_saved_round = 0

        # ======== channels ========
        self.connection_timeout = self.args.connection_timeout
        self.executors = None
        self.grpc_server = None

        # ======== Event Queue =======
        self.individual_client_events = {}  # Unicast
        self.sever_events_queue = collections.deque()
        self.broadcast_events_queue = collections.deque()  # Broadcast

        # ======== runtime information ========
        self.tasks_round = 0
        self.num_of_clients = 0

        # NOTE: sampled_participants = sampled_executors in deployment,
        # because every participant is an executor. However, in simulation mode,
        # executors is the physical machines (VMs), thus:
        # |sampled_executors| << |sampled_participants| as an VM may run multiple participants
        self.sampled_participants = []
        self.sampled_executors = []

        self.round_stragglers = []
        self.model_update_size = 0.

        self.collate_fn = None
        self.round = 0

        self.start_run_time = time.time()
        self.client_conf = {}

        self.stats_util_accumulator = []
        self.loss_accumulator = []
        self.client_training_results = []

        # number of registered executors
        self.registered_executor_info = set()
        self.test_result_accumulator = []
        self.testing_history = {'data_set': args.data_set, 'model': args.model, 'sample_mode': args.sample_mode,
                                'gradient_policy': args.gradient_policy, 'task': args.task,
                                'perf': collections.OrderedDict()}
        self.log_writer = SummaryWriter(log_dir=logger.logDir)
        #Faraz - dynamically set the per round total workers
        self.total_worker = args.total_worker if args.total_worker > 0 else args.initial_total_worker
        #Faraz - mode
        self.mode = self.args.mode
        logging.info(f"self.args.mode: {self.mode}")
        #Faraz - add clients dropout information
        self.clients_to_run = []
        self.clients_dropped_out = []
        self.dropped_clients_resource_usage = []
        self.deadline_differences = []
        self.deadline_differences_per_round = {}
        self.clients_dropped_out_per_round = {}
        self.dropped_clients_resource_usage_per_round = {}
        self.dropped_clients_resource_durations_per_round = {}
        #Faraz - FedBuff
        self.client_results_buffer = collections.deque()
        self.buffer_size = self.args.buffer_size or 10
        #Faraz - for dynamic network at clients
        self.bandwidth_profiles_dir = self.args.bandwidth_profiles_dir
        self.bandwidth_profiles = []
        #Faraz - client participation rate
        self.client_participation_rate = {}
        self.participation_per_round = {}
        self.clients_aggregated_per_round = {}
        self.clients_aggregated = []
        #Faraz - client resource usage
        self.total_resources_selected_clients = []
        self.total_virtual_time_selected_clients = []
        #Faraz - RL agent
        # self.rl_agent = RL()
        self.rl_agent = RL_singleQ()
        self.global_state = {}
        #optimizations
        self.optimizations = {}
        self.rl_updates = {}
        self.past_rl_updates = {}
        self.past_clients_dropped_out = []
        # except Exception as e:
        #     logging.error(e)
        #     raise e

        if args.wandb_token != "":
            os.environ['WANDB_API_KEY'] = args.wandb_token
            self.wandb = wandb
            if self.wandb.run is None:
                self.wandb.init(project=f'fedscale-{args.job_name}',
                                name=f'aggregator{args.this_rank}-{args.time_stamp}',
                                group=f'{args.time_stamp}')
                self.wandb.config.update({
                    "num_participants": args.num_participants,
                    "data_set": args.data_set,
                    "model": args.model,
                    "gradient_policy": args.gradient_policy,
                    "eval_interval": args.eval_interval,
                    "rounds": args.rounds,
                    "batch_size": args.batch_size,
                    "use_cuda": args.use_cuda
                })
            else:
                logging.error("Warning: wandb has already been initialized")
            # self.wandb.run.name = f'{args.job_name}-{args.time_stamp}'
        else:
            self.wandb = None

        # ======== Task specific ============
        self.init_task_context()

    def setup_env(self):
        """Set up experiments environment and server optimizer
        """
        self.setup_seed(seed=0)

    def setup_seed(self, seed=1):
        """Set global random seed for better reproducibility

        Args:
            seed (int): random seed

        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    
    def print_stats(self, clients_to_run):
        '''Faraz - Prints the stats of the current round'''
        try:
            logging.info(f"Round {self.round}:")
            logging.info(f"Wall clock time: {round(self.global_virtual_clock)}")
            logging.info(f"Selected participants to run: {clients_to_run}")
            logging.info("round: {}, clients aggregated: {}".format(self.round, self.clients_aggregated))
            logging.info("Number of clients aggregated: {}".format(len(self.clients_aggregated)))
            logging.info('Number of sampled participants: {}'.format(len(self.sampled_participants)))
            logging.info(f"Resource wastage of dropped clients: {self.dropped_clients_resource_usage_per_round[self.round]}")
            logging.info(f"Resource durations of dropped clients: {self.dropped_clients_resource_durations_per_round[self.round]}")
            logging.info(f"Number of clients dropped out: {len(self.clients_dropped_out_per_round[self.round])}")
            logging.info("round: {}, clients dropped out: {}".format(self.round, self.clients_dropped_out))
            logging.info("round: {}, stragglers: {}".format(self.round, self.round_stragglers))
            logging.info("round: {}, number of stragglers: {}".format(self.round, len(self.round_stragglers)))
            logging.info("round: {}, total resource usage of clients: {}".format(self.round, self.total_resources_selected_clients))
            logging.info('Real concurrent clients: {}'.format(len(clients_to_run) - len(self.clients_dropped_out_per_round[self.round])))
            logging.info('Virtual concurrent clients: {}'.format(len(clients_to_run) + len(self.clients_dropped_out_per_round[self.round])))
            logging.info(f"Deadline differences of clients: {self.deadline_differences_per_round[self.round]}")
            if len(self.deadline_differences_per_round[self.round]) > 0:
                logging.info(f"Mean of deadline differences: {np.mean(self.deadline_differences_per_round[self.round])}")
                logging.info(f"Std of deadline differences: {np.std(self.deadline_differences_per_round[self.round])}")
            logging.info(f"clients participation rates: {self.client_participation_rate}")
            logging.info(f"Participation per round: {self.participation_per_round[self.round]}")
            

            
            self.past_clients_dropped_out = copy.deepcopy(self.clients_dropped_out)
            self.clients_dropped_out = []
            self.total_resources_selected_clients = []
            self.clients_aggregated = []
            # logging.info('length of client results buffer: {}'.format(len(self.client_results_buffer)))
            # logging.info('length of sampled participants: {}'.format(len(self.clients_to_run)))
            #Faraz - find mean of deadline differences greater than and less than 0
            # logging.info(f"Mean of deadline differences of successful clients: {np.mean([x for x in self.deadline_differences_per_round[self.round] if x > 0])}")
            # logging.info(f"Mean of deadline differences of unsuccessful clients: {np.mean([x for x in self.deadline_differences_per_round[self.round] if x < 0])}")
        except Exception as e:
            logging.error(e)
            raise e
        
    #Faraz - clients dropout information update
    def update_client_dropped_out(self, client_id, exe_cost, deadline_difference=0, client_resources={}):
        try:
            # logging.info(f"Client {client_id} dropped out at round {self.round}")
            self.clients_dropped_out.append(client_id)
            self.dropped_clients_resource_usage.append({'clients': client_id, 'execution_durations': exe_cost, 'client_resources': client_resources})
            self.deadline_differences.append(deadline_difference)
            if self.round not in self.clients_dropped_out_per_round:
                self.clients_dropped_out_per_round[self.round] = []
                self.dropped_clients_resource_usage_per_round[self.round] = []
                self.dropped_clients_resource_durations_per_round[self.round] = []
                self.deadline_differences_per_round[self.round] = []
            self.deadline_differences_per_round[self.round].append(deadline_difference)
                
            self.clients_dropped_out_per_round[self.round].append(client_id)
            self.dropped_clients_resource_usage_per_round[self.round].append({'client': client_id, 'client_resources': client_resources})
            self.dropped_clients_resource_durations_per_round[self.round].append({'client': client_id, 'client_durations': exe_cost})
        except Exception as e:
            logging.error(e)
            raise e
        

    def init_control_communication(self):
        """Create communication channel between coordinator and executor.
        This channel serves control messages.
        """
        try:
            logging.info(f"Initiating control plane communication ...")
            if self.experiment_mode == commons.SIMULATION_MODE:
                num_of_executors = 0
                for ip_numgpu in self.args.executor_configs.split("="):
                    ip, numgpu = ip_numgpu.split(':')
                    for numexe in numgpu.strip()[1:-1].split(','):
                        for _ in range(int(numexe.strip())):
                            num_of_executors += 1
                self.executors = list(range(num_of_executors))
            else:
                self.executors = list(range(self.args.num_participants))

            # initiate a server process
            self.grpc_server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=20),
                options=[
                    ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                ],
            )
            job_api_pb2_grpc.add_JobServiceServicer_to_server(
                self, self.grpc_server)
            port = '[::]:{}'.format(self.args.ps_port)

            logging.info(f'%%%%%%%%%% Opening aggregator sever using port {port} %%%%%%%%%%')

            self.grpc_server.add_insecure_port(port)
            self.grpc_server.start()
        except Exception as e:
            logging.error(e)
            raise e

    def init_data_communication(self):
        """For jumbo traffics (e.g., training results).
        """
        pass

    def init_model(self):
        """Initialize the model"""
        if self.args.engine == commons.TENSORFLOW:
            self.model_wrapper = TensorflowModelAdapter(init_model())
        elif self.args.engine == commons.PYTORCH:
            self.model_wrapper = TorchModelAdapter(
                init_model(),
                optimizer=TorchServerOptimizer(
                    self.args.gradient_policy, self.args, self.device))
        else:
            raise ValueError(f"{self.args.engine} is not a supported engine.")
        self.model_weights = self.model_wrapper.get_weights()

    def init_task_context(self):
        """Initiate execution context for specific tasks
        """
        if self.args.task == "detection":
            cfg_from_file(self.args.cfg_file)
            np.random.seed(self.cfg.RNG_SEED)
            self.imdb, _, _, _ = combined_roidb(
                "voc_2007_test", ['DATA_DIR', self.args.data_dir], server=True)

    def init_client_manager(self, args):
        """ Initialize client sampler

        Args:
            args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py

        Returns:
            ClientManager: The client manager class

        Currently we implement two client managers:

        1. Random client sampler - it selects participants randomly in each round
        [Ref]: https://arxiv.org/abs/1902.01046

        2. Oort sampler
        Oort prioritizes the use of those clients who have both data that offers the greatest utility
        in improving model accuracy and the capability to run training quickly.
        [Ref]: https://www.usenix.org/conference/osdi21/presentation/lai

        """

        # sample_mode: random or oort
        client_manager = ClientManager(args.sample_mode, args=args)

        return client_manager

    def load_client_profile(self, file_path):
        """For Simulation Mode: load client profiles/traces

        Args:
            file_path (string): File path for the client profiles/traces

        Returns:
            dictionary: Return the client profiles/traces

        """
        logging.info(f"Loading client profiles")
        global_client_profile = {}
        if os.path.exists(file_path):
            with open(file_path, 'rb') as fin:
                # {client_id: [computer, bandwidth]}
                global_client_profile = pickle.load(fin)

        return global_client_profile

    def client_register_handler(self, executorId, info):
        """Triggered once receive new executor registration.

        Args:
            executorId (int): Executor Id
            info (dictionary): Executor information

        """
        #Faraz - Adding dynamic network profiles per round
        try:
            logging.info(f"Loading {len(info['size'])} client traces ...")
            
            logging.info(f"Reading bandwidth profiles from {self.bandwidth_profiles_dir}")
            
            
            
            ##########################Compute and network clustering#########################


            # if self.bandwidth_profiles_dir != '':
            #     dynamic_network_profiles = pd.read_csv(self.bandwidth_profiles_dir)
            #     # Normalize the throughput values if necessary
            #     max_throughput = dynamic_network_profiles['throughput_mbps'].max()
            #     dynamic_network_profiles['normalized_throughput'] = dynamic_network_profiles['throughput_mbps'] / max_throughput

            #     # Perform K-means clustering on bandwidth profiles
            #     k = 3  # Number of bandwidth clusters
            #     kmeans = KMeans(n_clusters=k)
            #     kmeans.fit(dynamic_network_profiles[['normalized_throughput']].values)

            #     # Assign each bandwidth profile to its corresponding cluster
            #     cluster_labels = kmeans.labels_
            #     dynamic_network_profiles['cluster'] = cluster_labels

            #     # Sort the bandwidth profiles within each cluster based on the normalized throughput
            #     dynamic_network_profiles = dynamic_network_profiles.sort_values(['cluster', 'normalized_throughput'], ascending=[True, False])

            #     # Retrieve the bandwidth profiles for each cluster
            #     clusters = []
            #     for cluster_id in range(k):
            #         cluster_profiles = dynamic_network_profiles[dynamic_network_profiles['cluster'] == cluster_id]
            #         cluster_bandwidths = list(cluster_profiles['throughput_mbps'] * 1024)  # Convert to kbps
            #         clusters.append(cluster_bandwidths)

            # # Perform client profiles clustering
            # client_profiles_df = pd.DataFrame(self.client_profiles).T
            # computation_values = client_profiles_df['computation'].values.reshape(-1, 1)

            # k = 3  # Number of client profiles clusters
            # kmeans = KMeans(n_clusters=k)
            # kmeans.fit(computation_values)
            # client_profiles_df['computation_cluster'] = kmeans.labels_

            # # Sort client profiles by the cluster labels
            # client_profiles_df = client_profiles_df.sort_values('computation_cluster')

            # # Combine bandwidth and client profiles clustering
            # clustered_bandwidths = []
            # clustered_system_profiles = []
            # for cluster_id in range(k):
            #     cluster_bandwidths = clusters[cluster_id]
            #     cluster_profiles = client_profiles_df[client_profiles_df['computation_cluster'] == cluster_id]
            #     cluster_size = len(cluster_profiles)
            #     clustered_bandwidths.extend(cluster_bandwidths[:cluster_size])
            #     clustered_system_profiles.extend([self.client_profiles.get(mapped_id, {'computation': 1.0, 'communication': 1.0}) for mapped_id in cluster_profiles.index])

            # logging.info(f"Clustered bandwidths: {clustered_bandwidths}")
            # logging.info(f"Clustered system profiles: {clustered_system_profiles}")
            # for i, _size in enumerate(info['size']):
            #     cluster_id = i % k  # Map size to cluster ID
            #     cluster_bandwidths = clustered_bandwidths[cluster_id]
            #     cluster_system_profiles = clustered_system_profiles[cluster_id]

            #     client_id = (self.num_of_clients + 1) if self.experiment_mode == commons.SIMULATION_MODE else executorId
            #     if client_id not in self.client_participation_rate:
            #         self.client_participation_rate[client_id] = 0
            #     self.client_manager.register_client(
            #         executorId, client_id, size=_size, speed=cluster_system_profiles, possible_bandwidths=cluster_bandwidths)
            #     self.client_manager.registerDuration(
            #         client_id,
            #         batch_size=self.args.batch_size,
            #         local_steps=self.args.local_steps,
            #         upload_size=self.model_update_size,
            #         download_size=self.model_update_size
            #     )
            #     self.num_of_clients += 1


            
            ##########################Network clustering#########################
            clusters = []
            if self.bandwidth_profiles_dir != '':
                dynamic_network_profiles = pd.read_csv(self.bandwidth_profiles_dir)
                # Normalize the throughput values if necessary
                max_throughput = dynamic_network_profiles['throughput_mbps'].max()
                dynamic_network_profiles['normalized_throughput'] = dynamic_network_profiles['throughput_mbps'] / max_throughput

                # Perform K-means clustering
                k = 2  # Number of clusters
                kmeans = KMeans(n_clusters=k)
                kmeans.fit(dynamic_network_profiles[['normalized_throughput']].values)

                # Assign each bandwidth profile to its corresponding cluster
                cluster_labels = kmeans.labels_
                dynamic_network_profiles['cluster'] = cluster_labels

                # Sort the bandwidth profiles within each cluster based on the normalized throughput
                dynamic_network_profiles = dynamic_network_profiles.sort_values(['cluster', 'normalized_throughput'], ascending=[True, False])

                # Retrieve the bandwidth profiles for each cluster
                for cluster_id in range(k):
                    cluster_profiles = dynamic_network_profiles[dynamic_network_profiles['cluster'] == cluster_id]
                    cluster_bandwidths = list(cluster_profiles['throughput_mbps'] * 1024)  # Convert to kbps
                    clusters.append(cluster_bandwidths)
            
            logging.info(f"clusters bandwidths: {clusters}")
            for _size in info['size']:
                # cluster_id = (_size - 1) % k  # Map size to cluster ID
                #high network
                cluster_id = 0 # Map size to cluster ID
                cluster_bandwidths = clusters[cluster_id]

                # Since the worker rankId starts from 1, we also configure the initial dataId as 1
                mapped_id = (self.num_of_clients + 1) % len(self.client_profiles) if len(self.client_profiles) > 0 else 1
                systemProfile = self.client_profiles.get(mapped_id, {'computation': 1.0, 'communication': 1.0})

                client_id = (self.num_of_clients + 1) if self.experiment_mode == commons.SIMULATION_MODE else executorId
                if client_id not in self.client_participation_rate:
                    self.client_participation_rate[client_id] = 0
                self.client_manager.register_client(
                    executorId, client_id, size=_size, speed=systemProfile, possible_bandwidths=cluster_bandwidths)
                self.client_manager.registerDuration(
                    client_id,
                    batch_size=self.args.batch_size,
                    local_steps=self.args.local_steps,
                    upload_size=self.model_update_size,
                    download_size=self.model_update_size
                )
                self.num_of_clients += 1
                        ###################################################
                        
            #Faraz - Read the CSV file into a pandas DataFrame
            # if self.bandwidth_profiles_dir!='':
            #     dynamic_network_profiles = pd.read_csv(self.bandwidth_profiles_dir)
            #     #Faraz - Iterate over the dataset and update throughputs for num_tcp_conn = 8
            #     for index, row in dynamic_network_profiles.iterrows():
            #         if row['num_tcp_conn'] > 1:
            #             dynamic_network_profiles.at[index, 'throughput_mbps'] /= int(row['num_tcp_conn'])
            #     self.bandwidth_profiles = list(dynamic_network_profiles["throughput_mbps"]*1024) #Faraz - Convert to kbps
                
            # logging.info(f"Dynamic bandwidth profiles: {self.bandwidth_profiles}")
            
            # for _size in info['size']:
            #     # since the worker rankId starts from 1, we also configure the initial dataId as 1
            #     mapped_id = (self.num_of_clients + 1) % len(
            #         self.client_profiles) if len(self.client_profiles) > 0 else 1
            #     systemProfile = self.client_profiles.get(
            #         mapped_id, {'computation': 1.0, 'communication': 1.0})


            #     client_id = (
            #             self.num_of_clients + 1) if self.experiment_mode == commons.SIMULATION_MODE else executorId
            #     #Faraz - Add client_id to client_participation_rate
            #     if client_id not in self.client_participation_rate:
            #         self.client_participation_rate[client_id] = 0
            #     self.client_manager.register_client(
            #         executorId, client_id, size=_size, speed=systemProfile, possible_bandwidths=self.bandwidth_profiles)
            #     self.client_manager.registerDuration(
            #         client_id,
            #         batch_size=self.args.batch_size,
            #         local_steps=self.args.local_steps,
            #         upload_size=self.model_update_size,
            #         download_size=self.model_update_size
            #     )
            #     self.num_of_clients += 1

            logging.info("Info of all feasible clients {}".format(
                self.client_manager.getDataInfo()))
        except Exception as e:
            logging.error('Error in registering clients: ', e)
            raise e

    def executor_info_handler(self, executorId, info):
        """Handler for register executor info and it will start the round after number of
        executor reaches requirement.

        Args:
            executorId (int): Executor Id
            info (dictionary): Executor information

        """
        self.registered_executor_info.add(executorId)
        logging.info(
            f"Received executor {executorId} information, {len(self.registered_executor_info)}/{len(self.executors)}")

        # In this simulation, we run data split on each worker, so collecting info from one executor is enough
        # Waiting for data information from executors, or timeout
        if self.experiment_mode == commons.SIMULATION_MODE:

            if len(self.registered_executor_info) == len(self.executors):
                self.client_register_handler(executorId, info)
                # start to sample clients
                self.round_completion_handler()
        else:
            # In real deployments, we need to register for each client
            self.client_register_handler(executorId, info)
            if len(self.registered_executor_info) == len(self.executors):
                self.round_completion_handler()

    def get_optimization(self, client_to_run):
        '''Get the optimization method for the current client.'''
        client_local_state = self.client_manager.get_client_local_state(client_to_run)
        optimization = self.rl_agent.choose_action_per_client(self.global_state, client_local_state, client_to_run)
        return optimization
    
    def compress_model(self, optimization):
        try:
            compressed_weights = []
            total_size = 0
            q_compressor = QSGDCompressor(n_bit=optimization, random=True, cuda=False)
            # logging.info('HERE41')
            for weight in self.model_wrapper.get_weights():
                compressed_weight, size = q_compressor.compress(weight)
                # logging.info('HERE42')
                total_size += int(size)
                # logging.info('HERE42')
                compressed_weights.append(compressed_weight)
                # logging.info('HERE43')
                
            return compressed_weights, total_size
        except Exception as e:
            logging.error('Error in compressing model: ', e)
            raise e
    
    def decompress_model(self, model, optimization=8):
        try:
            decompressed_weights = []
            q_compressor = QSGDCompressor(n_bit=optimization, random=True, cuda=False)
            for weight in model:
                decompressed_weight = q_compressor.decompress(weight)
                decompressed_weights.append(decompressed_weight)
            return decompressed_weights
        except Exception as e:
            logging.error('Error in decompressing model: ', e)
            raise e
    def update_RL_agent(self):
        '''Update the RL agent with the current client's information.'''
        try:
            logging.info('Updating RL agent')
            for client_id, update in self.rl_updates.items():
                if 'global_state' in update:
                    global_state = update['global_state']
                    local_state = update['local_state']
                    optimization = update['optimization']
                    new_global_state = update['new_global_state']
                    new_local_state = update['new_local_state']
                    reward = update['reward']
                    reward['accuracy'] = np.mean(reward['accuracy'])
                    self.rl_agent.update_Q_per_client(client_id, global_state, local_state, optimization, new_global_state, new_local_state, reward)
                    RL_path = '${FLOAT_HOME}/benchmark/logs/rl_model'
                    logging.info(f'Updated RL Q table path: {RL_path}')
                    self.rl_agent.save_Q(RL_path)
                    # logging.info(f'Updated RL Q table: {self.rl_agent.Q}')
                else:
                    logging.info('No update for RL agent')
            # logging.info('update_RL_agent: rl_updates: {}'.format(self.rl_updates))
            
            # self.rl_agent.print_overhead_times()
        except Exception as e:
            logging.error('Error in updating RL agent: ', e)
            raise e

    def perform_optimization(self, client_cfg, client_to_run, optimization, oldroundDuration):
        '''Perform the optimization method for the current client.'''
        try:
            client_local_state = self.client_manager.get_client_local_state(client_to_run)
            # logging.info('HERE40')
            compressed_weights, size = self.compress_model(optimization)
            logging.info(f"Faraz - Compressed model size: {size / 1024.0 * 8}, seize before compression: {self.model_update_size}")
            size =  size / 1024.0 * 8.  # kbits
            exe_cost = self.client_manager.get_completion_time(
                        client_to_run,
                        batch_size=client_cfg.batch_size,
                        local_steps=client_cfg.local_steps,
                        upload_size=size,
                        download_size=self.model_update_size,
                        variable_resources = True)
            
            roundDuration = exe_cost['computation'] + \
                                        exe_cost['communication']

            isactivewithouttraining, olddeadline_differencewithouttraining = self.client_manager.isClientActivewithDeadline(client_to_run, self.global_virtual_clock)
            isactive, olddeadline_difference = self.client_manager.isClientActivewithDeadline(client_to_run, oldroundDuration + self.global_virtual_clock)
            client_active, deadline_difference = self.client_manager.isClientActivewithDeadline(client_to_run, roundDuration + self.global_virtual_clock)
            logging.info(f"Faraz - Client {client_to_run} is active: {client_active}, deadline difference: {deadline_difference}, old deadline difference: {olddeadline_difference}, old deadline difference without training: {olddeadline_differencewithouttraining}, round duration difference: {oldroundDuration - roundDuration}")
            if client_active:
                logging.info('Faraz - Successfully scheduled client {} for round {} with optimization {} and round duration reduction of {}%'.format(client_to_run, self.round, optimization, oldroundDuration - roundDuration))
                
                # self.rl_agent.update_Q_per_client(client_to_run, self.global_state, client_local_state, optimization, self.global_state, client_local_state, 1)
                self.rl_updates[client_to_run] = {'client_to_run': client_to_run, 'global_state': self.global_state, 'local_state': client_local_state, 'optimization': optimization, 'new_global_state': self.global_state, 'new_local_state': client_local_state, 'reward': {'participation_success': 1, 'accuracy': []}}
                logging.info('rl_updates: {}'.format(self.rl_updates))
                self.rl_agent.print_overhead_times()
                return True, roundDuration, compressed_weights, exe_cost
            else:
                logging.info('Faraz - Failed to schedule client {} for round {} with optimization {} and round duration reduction of {}%'.format(client_to_run, self.round, optimization, oldroundDuration - roundDuration))
                # self.rl_agent.update_Q_per_client(client_to_run, self.global_state, client_local_state, optimization, self.global_state, client_local_state, -1)
                self.rl_updates[client_to_run] =  {'client_to_run': client_to_run, 'global_state': self.global_state, 'local_state': client_local_state, 'optimization': optimization, 'new_global_state': self.global_state, 'new_local_state': client_local_state, 'reward': {'participation_success': -1, 'accuracy': []}}
                logging.info('rl_updates: {}'.format(self.rl_updates))
                self.rl_agent.print_overhead_times()
                return False, roundDuration, compressed_weights, exe_cost
        except Exception as e:
            logging.error('Error in performing optimization: ', e)
            raise e
            
        
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
                    download_size=self.model_update_size)

                    roundDuration = exe_cost['computation'] + \
                                    exe_cost['communication']
                    client_active, deadline_difference = self.client_manager.isClientActivewithDeadline(client_to_run, roundDuration + self.global_virtual_clock)
                    # newRoundDuration, compressed_weights, new_exe_cost = roundDuration, None, None
                    # action = 0
                    
                    if client_active:
                        sampledClientsReal.append(client_to_run)
                        completionTimes.append(roundDuration)
                        completed_client_clock[client_to_run] = exe_cost
                        client_completion_times[client_to_run] = roundDuration
                        client_resources[client_to_run] = exe_cost
                    else:
                        clients_left_out.append(client_to_run)
                        sampledClientsReal.append(client_to_run)
                        completionTimes.append(roundDuration)
                        completed_client_clock[client_to_run] = exe_cost
                        client_completion_times[client_to_run] = roundDuration
                        client_resources[client_to_run] = exe_cost
                    # if not client_active:
                    #     action = self.get_optimization(client_to_run)
                    #     # logging.info('action is {}'.format(action))
                    #     client_active, newRoundDuration, compressed_weights, new_exe_cost = self.perform_optimization(client_cfg, client_to_run, action, roundDuration)
                    # logging.info('tictak clients: self.rl_updates: {}'.format(self.rl_updates))
                    # if the client is not active by the time of collection, we consider it is lost in this round
                    
                    # if client_active and new_exe_cost:
                    #     sampledClientsReal.append(client_to_run)
                    #     completionTimes.append(newRoundDuration)
                    #     self.optimizations[client_to_run] = {'optimization': action, 'model_weights': compressed_weights}
                    #     completed_client_clock[client_to_run] = new_exe_cost
                    #     client_completion_times[client_to_run] = newRoundDuration
                    #     client_resources[client_to_run] = new_exe_cost
                    # elif client_active:
                    #     sampledClientsReal.append(client_to_run)
                    #     completionTimes.append(roundDuration)
                    #     completed_client_clock[client_to_run] = exe_cost
                    #     client_completion_times[client_to_run] = roundDuration
                    #     client_resources[client_to_run] = exe_cost
                    # else:
                    #     clients_left_out.append(client_to_run)
                    #     sampledClientsReal.append(client_to_run)
                    #     completionTimes.append(roundDuration)
                    #     completed_client_clock[client_to_run] = exe_cost
                    #     client_completion_times[client_to_run] = roundDuration
                    #     client_resources[client_to_run] = exe_cost
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
                clients_to_run = real_clients_to_run
                 
                stragglers = [sampledClientsReal[k]
                            for k in workers_sorted_by_completion_time[num_clients_to_collect:]]
                
                max_time = 0
                if len(clients_to_run) > 0:
                    for client in clients_to_run:
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

    def _is_first_result_in_round(self):
        return self.model_in_update == 1

    def _is_last_result_in_round(self):
        return self.model_in_update == self.tasks_round

    def select_participants(self, select_num_participants, overcommitment=1.0, sort_clients = False):
        """Select clients for next round.

        Args:
            select_num_participants (int): Number of clients to select.
            overcommitment (float): Overcommit ration for next round.

        Returns:
            list of int: The list of sampled clients id.

        """
        try:
            if self.mode == 'fedAvg':
                return self.client_manager.select_participants(
                    select_num_participants,
                    cur_time=self.global_virtual_clock,
                )
            if sort_clients:
                return sorted(self.client_manager.select_participants(
                    int(select_num_participants * overcommitment),
                    cur_time=self.global_virtual_clock),
                )
            else:
                return self.client_manager.select_participants(
                    int(select_num_participants * overcommitment),
                    cur_time=self.global_virtual_clock,
                )
        except Exception as e:
            logging.error('Error in selecting participants: ', e)
            raise e

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

            self.model_in_update += 1
            self.update_weight_aggregation(results)

            self.update_lock.release()
        except Exception as e:
            logging.error('Error in client completion handler: ', e)
            raise e

    def update_weight_aggregation(self, results):
        """Updates the aggregation with the new results.

        :param results: the results collected from the client.
        """
        try:
            # logging.info(f'update_weight_aggregation: {results}')
            update_weights = results['update_weight']
            if results.get('optimization'):
                # logging.info(f'type of update_weights: {type(update_weights)}')
                # logging.info(f'update_weights: {(update_weights)}')
                update_weights = self.decompress_model(update_weights)
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


    def aggregate_test_result(self):
        try:
            accumulator = self.test_result_accumulator[0]
            for i in range(1, len(self.test_result_accumulator)):
                if self.args.task == "detection":
                    for key in accumulator:
                        if key == "boxes":
                            for j in range(596):
                                accumulator[key][j] = accumulator[key][j] + \
                                                    self.test_result_accumulator[i][key][j]
                        else:
                            accumulator[key] += self.test_result_accumulator[i][key]
                else:
                    for key in accumulator:
                        accumulator[key] += self.test_result_accumulator[i][key]
            self.testing_history['perf'][self.round] = {'round': self.round, 'clock': self.global_virtual_clock}
            for metric_name in accumulator.keys():
                if metric_name == 'test_loss':
                    self.testing_history['perf'][self.round]['loss'] = accumulator['test_loss'] \
                        if self.args.task == "detection" else accumulator['test_loss'] / accumulator['test_len']
                elif metric_name not in ['test_len']:
                    self.testing_history['perf'][self.round][metric_name] \
                        = accumulator[metric_name] / accumulator['test_len']

            round_perf = self.testing_history['perf'][self.round]
            logging.info(
                "FL Testing in round: {}, virtual_clock: {}, results: {}"
                .format(self.round, self.global_virtual_clock, round_perf))
        except Exception as e:
            logging.error('Error in aggregate test result: ', e)
            raise e

    def update_default_task_config(self):
        """Update the default task configuration after each round
        """
        if self.round % self.args.decay_round == 0:
            self.args.learning_rate = max(
                self.args.learning_rate * self.args.decay_factor, self.args.min_learning_rate)

    def round_completion_handler(self):
        """Triggered upon the round completion, it registers the last round execution info,
        broadcast new tasks for executors and select clients for next round.
        """
        try:
            # logging.info("HERE3")
            self.global_virtual_clock += self.round_duration
            self.round += 1
            # logging.info('round_completion_handler1: rl_updates: {}'.format(self.rl_updates))
            # if self.rl_updates != {}:
            #     self.update_RL_agent()
            #     #Faraz- Reset the updates
            #     self.rl_updates = {}
            #     gc.collect()
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
                if self.round % 50 == 0 or self.round == 2:
                    self.broadcast_aggregator_events(commons.UPDATE_MODEL)
                    self.broadcast_aggregator_events(commons.MODEL_TEST)
                    # self.rl_agent.save_Q('rl_agent')
                    if self.round % 50 == 0 or self.round == 2:
                        self.broadcast_aggregator_events(commons.CLIENT_VALIDATE_ALL)
                else:
                    self.broadcast_aggregator_events(commons.UPDATE_MODEL)
                    self.broadcast_aggregator_events(commons.START_ROUND)
                if self.rl_updates!={}:
                    # logging.info('Going in rl_updates: {}'.format(self.rl_updates))
                    self.broadcast_aggregator_events(commons.UPDATE_MODEL)
                    self.broadcast_aggregator_events(commons.CLIENT_VALIDATE)
                    
                
                
            else:
                #Faraz - Skip round if no clients to run
                self.round_completion_handler()
                logging.info('No clients to run, skipping round')
        except Exception as e:
            logging.error('Error in round completion handler: ', e)
            raise e

    def log_train_result(self, avg_loss):
        try:
            """Log training result on TensorBoard and optionally WanDB
            """
            self.log_writer.add_scalar('Train/round_to_loss', avg_loss, self.round)
            self.log_writer.add_scalar(
                'Train/time_to_train_loss (min)', avg_loss, self.global_virtual_clock / 60.)
            self.log_writer.add_scalar(
                'Train/round_duration (min)', self.round_duration / 60., self.round)
            self.log_writer.add_histogram(
                'Train/client_duration (min)', self.flatten_client_duration, self.round)

            if self.wandb != None:
                self.wandb.log({
                    'Train/round_to_loss': avg_loss,
                    'Train/round_duration (min)': self.round_duration/60.,
                    'Train/client_duration (min)': self.flatten_client_duration,
                    'Train/time_to_round (min)': self.global_virtual_clock/60.,
                }, step=self.round)
        except Exception as e:
            logging.error('Error in log train result: ', e)
            raise e
        
    def log_test_result(self):
        """Log testing result on TensorBoard and optionally WanDB
        """
        try:
            self.log_writer.add_scalar(
                'Test/round_to_loss', self.testing_history['perf'][self.round]['loss'], self.round)
            self.log_writer.add_scalar(
                'Test/round_to_accuracy', self.testing_history['perf'][self.round]['top_1'], self.round)
            self.log_writer.add_scalar('Test/time_to_test_loss (min)', self.testing_history['perf'][self.round]['loss'],
                                    self.global_virtual_clock / 60.)
            self.log_writer.add_scalar('Test/time_to_test_accuracy (min)', self.testing_history['perf'][self.round]['top_1'],
                                    self.global_virtual_clock / 60.)
        except Exception as e:
            logging.error('Error in log test result: ', e)
            raise e

    def save_model(self):
        """Save model to the wandb server if enabled
        
        """
        logging.info('Saving model')
        if parser.args.save_checkpoint and self.last_saved_round < self.round:
            self.last_saved_round = self.round
            np.save(self.temp_model_path, self.model_weights)
            if self.wandb != None:
                artifact = self.wandb.Artifact(name='model_'+str(self.this_rank), type='model')
                artifact.add_file(local_path=self.temp_model_path)
                self.wandb.log_artifact(artifact)

    def deserialize_response(self, responses):
        """Deserialize the response from executor

        Args:
            responses (byte stream): Serialized response from executor.

        Returns:
            string, bool, or bytes: The deserialized response object from executor.
        """
        return pickle.loads(responses)

    def serialize_response(self, responses):
        """ Serialize the response to send to server upon assigned job completion

        Args:
            responses (ServerResponse): Serialized response from server.

        Returns:
            bytes: The serialized response object to server.

        """
        return pickle.dumps(responses)

    # def validation_completion_handler(self, client_id, results):
    #     """Each executor will handle a subset of validation dataset

    #     Args:
    #         client_id (int): The client id.
    #         results (dictionary): The client validation accuracies.

    #     """
    #     try:
    #         logging.info('Validation completion handler results: {}'.format(results))
    #         # logging.info('validation_completion_handler: BEFORE self.rl_updates: {}'.format(self.rl_updates))
    #         for client_id, accuracy in results.items():
    #             # logging.info('Client {} validation accuracy: {}'.format(client_id, accuracy))
    #             if client_id not in self.rl_updates:
    #                 if 'reward' not in self.rl_updates[client_id]:
    #                     self.rl_updates[client_id]['reward'] = {}
    #                     self.rl_updates[client_id]['reward']['accuracy'] = []
    #             self.rl_updates[client_id]['reward']['accuracy'].append(float(("%.17f" % accuracy).rstrip('0').rstrip('.')))
            
    #         # logging.info('validation_completion_handler: AFTER self.rl_updates: {}'.format(self.rl_updates))

    #     except Exception as e:
    #         logging.error('Error in validation completion handler: ', e)
    #         raise e
        
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
            
    def testing_completion_handler(self, client_id, results):
        
        """Each executor will handle a subset of testing dataset

        Args:
            client_id (int): The client id.
            results (dictionary): The client test results.

        """
        try:
            results = results['results']

            # List append is thread-safe
            self.test_result_accumulator.append(results)

            # Have collected all testing results

            if len(self.test_result_accumulator) == len(self.executors):

                self.aggregate_test_result()
                # Dump the testing result
                with open(os.path.join(logger.logDir, 'testing_perf'), 'wb') as fout:
                    pickle.dump(self.testing_history, fout)

                self.save_model()

                if len(self.loss_accumulator):
                    logging.info("logging test result")
                    self.log_test_result()

                self.broadcast_events_queue.append(commons.START_ROUND)
        except Exception as e:
            logging.error('Error in testing completion handler: ', e)
            raise e

    def broadcast_aggregator_events(self, event):
        """Issue tasks (events) to aggregator worker processes by adding grpc request event
        (e.g. MODEL_TEST, MODEL_TRAIN) to event_queue.

        Args:
            event (string): grpc event (e.g. MODEL_TEST, MODEL_TRAIN) to event_queue.

        """
        self.broadcast_events_queue.append(event)

    def dispatch_client_events(self, event, clients=None):
        """Issue tasks (events) to clients

        Args:
            event (string): grpc event (e.g. MODEL_TEST, MODEL_TRAIN) to event_queue.
            clients (list of int): target client ids for event.

        """
        try:
            # logging.info('Faraz - clients: {} have been dispatched event: {}'.format(clients, event))
            if clients is None:
                clients = self.sampled_executors

            for client_id in clients:
                self.individual_client_events[client_id].append(event)
            # logging.info(f'self.individual_client_events: {self.individual_client_events}')
            # logging.info('Faraz - clients: {} have been dispatched event: {}'.format(clients, event))
            
        except Exception as e:
            logging.error('Error in dispatch client events: ', e)
            raise e

    def get_client_conf(self, client_id):
        """Training configurations that will be applied on clients,
        # developers can further define personalized client config here.

        Args:
            client_id (int): The client id.

        Returns:
            dictionary: TorchClient training config.

        """
        conf = {
            'learning_rate': self.args.learning_rate,
        }
        return conf

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

    def get_val_config(self, clients=None):
        """FL model testing on clients, developers can further define personalized client config here.

        Args:
            client_id (int): The client id.

        Returns:
            dictionary: The testing config for new task.

        """
        # logging.info('get_val_config self.rl_updates: {}'.format(self.rl_updates))
        if clients:
            return {'round': self.round, 'clients': clients}
        return {'round': self.round, 'clients': self.client_manager.getAllClients()}
    
    def get_test_config(self, client_id):
        """FL model testing on clients, developers can further define personalized client config here.

        Args:
            client_id (int): The client id.

        Returns:
            dictionary: The testing config for new task.

        """
        return {'client_id': client_id}

    def get_shutdown_config(self, client_id):
        """Shutdown config for client, developers can further define personalized client config here.

        Args:
            client_id (int): TorchClient id.

        Returns:
            dictionary: Shutdown config for new task.

        """
        return {'client_id': client_id}

    def add_event_handler(self, client_id, event, meta, data):
        """ Due to the large volume of requests, we will put all events into a queue first.

        Args:
            client_id (int): The client id.
            event (string): grpc event MODEL_TEST or UPLOAD_MODEL.
            meta (dictionary or string): Meta message for grpc communication, could be event.
            data (dictionary): Data transferred in grpc communication, could be model parameters, test result.

        """
        self.sever_events_queue.append((client_id, event, meta, data))

    def CLIENT_REGISTER(self, request, context):
        """FL TorchClient register to the aggregator

        Args:
            request (RegisterRequest): Registeration request info from executor.

        Returns:
            ServerResponse: Server response to registeration request

        """

        try:
            # NOTE: client_id = executor_id in deployment,
            # while multiple client_id uses the same executor_id (VMs) in simulations
            # logging.info('Faraz - Registring client')
            executor_id = request.executor_id
            executor_info = self.deserialize_response(request.executor_info)
            if executor_id not in self.individual_client_events:
                # logging.info(f"Detect new client: {executor_id}, executor info: {executor_info}")
                self.individual_client_events[executor_id] = collections.deque()
            else:
                logging.info(f"Previous client: {executor_id} resumes connecting")

            # logging.info(f"self.individual_client_events: {self.individual_client_events}")
            #We can customize whether to admit the clients here
            self.executor_info_handler(executor_id, executor_info)
            dummy_data = self.serialize_response(commons.DUMMY_RESPONSE)

            return job_api_pb2.ServerResponse(event=commons.DUMMY_EVENT,
                                            meta=dummy_data, data=dummy_data)
        except Exception as e:
            logging.error('Error in client registeration handler: ', e)
            raise e

    def CLIENT_PING(self, request, context):
        """Handle client ping requests

        Args:
            request (PingRequest): Ping request info from executor.

        Returns:
            ServerResponse: Server response to ping request

        """
        try:
            # logging.info(f"Receive ping request from client: {request}")
            # NOTE: client_id = executor_id in deployment,
            # while multiple client_id may use the same executor_id (VMs) in simulations
            executor_id, client_id = request.executor_id, request.client_id
            response_data = response_msg = commons.DUMMY_RESPONSE

            if len(self.individual_client_events[executor_id]) == 0:
                # send dummy response
                current_event = commons.DUMMY_EVENT
                response_data = response_msg = commons.DUMMY_RESPONSE
            else:
                current_event = self.individual_client_events[executor_id].popleft()
                if current_event == commons.CLIENT_TRAIN:
                    response_msg, response_data = self.create_client_task(
                        executor_id)
                    if response_msg is None:
                        current_event = commons.DUMMY_EVENT
                        if self.experiment_mode != commons.SIMULATION_MODE:
                            self.individual_client_events[executor_id].append(
                                commons.CLIENT_TRAIN)
                    # logging.info(f'len(self.stats_util_accumulator), self.tasks_round-1: {len(self.stats_util_accumulator)}, {len(self.clients_to_run)-1}')
                    # if self.rl_updates != {} and len(self.stats_util_accumulator) >= len(self.clients_to_run)-2:
                    #     logging.info('broadcasting client validate model')
                    #     self.broadcast_aggregator_events(commons.UPDATE_MODEL)
                    #     self.broadcast_aggregator_events(commons.CLIENT_VALIDATE)
                elif current_event == commons.CLIENT_VALIDATE_ALL:
                    response_msg = self.get_test_config(client_id)
                    response_data = self.get_val_config()
                elif current_event == commons.CLIENT_VALIDATE:
                    response_msg = self.get_test_config(client_id)
                    response_data = self.get_val_config(clients=list(self.rl_updates.keys()))
                elif current_event == commons.MODEL_TEST:
                    response_msg = self.get_test_config(client_id)
                elif current_event == commons.UPDATE_MODEL:
                    response_data = self.model_wrapper.get_weights()
                elif current_event == commons.SHUT_DOWN:
                    response_msg = self.get_shutdown_config(executor_id)

            response_msg, response_data = self.serialize_response(
                response_msg), self.serialize_response(response_data)
            # NOTE: in simulation mode, response data is pickle for faster (de)serialization
            response = job_api_pb2.ServerResponse(event=current_event,
                                                meta=response_msg, data=response_data)
            if current_event != commons.DUMMY_EVENT:
                logging.info(f"Issue EVENT ({current_event}) to EXECUTOR ({executor_id})")
        except Exception as e:
            logging.error(f"Error in CLIENT_PING: {e}")
            # response = job_api_pb2.ServerResponse(event=commons.DUMMY_EVENT,
                                                # meta=commons.DUMMY_RESPONSE, data=commons.DUMMY_RESPONSE)

        return response

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

                        self.individual_client_events[executor_id].append(
                            commons.CLIENT_TRAIN)

            elif event in (commons.MODEL_TEST, commons.UPLOAD_MODEL, commons.CLIENT_VALIDATE, commons.CLIENT_VALIDATE_ALL):
                self.add_event_handler(
                    executor_id, event, meta_result, data_result)
            else:
                logging.error(f"Received undefined event {event} from client {client_id}")

            return self.CLIENT_PING(request, context)
        except Exception as e:
            logging.error(f"Error in CLIENT_EXECUTE_COMPLETION: {e}")
            raise e

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

    def stop(self):
        """Stop the aggregator
        """
        logging.info(f"Terminating the aggregator ...")
        if self.wandb != None:
            self.wandb.finish()
        time.sleep(5)

if __name__ == "__main__":
    try:
        print("Faraz - Starting aggregator ...")
        aggregator = Aggregator(parser.args)
        aggregator.run()
    except Exception as e:
        logging.error(f"Error in aggregator: {e}")
        raise e
