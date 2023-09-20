# -*- coding: utf-8 -*-
import collections
import gc
import pickle
import random
import time
from argparse import Namespace

import numpy as np
import torch
import wandb

import fedscale.cloud.channels.job_api_pb2 as job_api_pb2
import fedscale.cloud.logger.executor_logging as logger
from fedscale.cloud.channels.channel_context import ClientConnections
from fedscale.cloud.execution.tensorflow_client import TensorflowClient
from fedscale.cloud.execution.torch_client import TorchClient
from fedscale.cloud.execution.torch_client_quantized import TorchClientQuantized
from fedscale.cloud.execution.data_processor import collate, voice_collate_fn
from fedscale.cloud.execution.rl_client import RLClient
from fedscale.cloud.fllibs import *
from fedscale.dataloaders.divide_data import DataPartitioner, select_dataset
from fedscale.utils.compressors.quantization import QSGDCompressor
from fedscale.utils.compressors.pruning import Pruning
import traceback

class Executor(object):
    """Abstract class for FedScale executor.

    Args:
        args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py

    """

    def __init__(self, args):
        # initiate the executor log path, and executor ips
        try:
            logger.initiate_client_setting()

            self.model_adapter = self.get_client_trainer(args).get_model_adapter(init_model())
            self.training_requests_received = 0
            self.args = args
            self.num_executors = args.num_executors
            # ======== env information ========
            self.this_rank = args.this_rank
            self.executor_id = str(self.this_rank)

            # ======== model and data ========
            self.training_sets = self.test_dataset = None

            # ======== channels ========
            self.aggregator_communicator = ClientConnections(
                args.ps_ip, args.ps_port)

            # ======== runtime information ========
            self.collate_fn = None
            self.round = 0
            self.start_run_time = time.time()
            self.received_stop_request = False
            self.event_queue = collections.deque()

            self.client_accuracies = {}
            self.client_losses = {}
            self.old_client_losses = {}
            self.old_client_accuracies = {}
            #Faraz - optimizations
            self.q_compressor = QSGDCompressor(random=True, cuda=False)
            self.pruning = Pruning()
            
            if args.wandb_token != "":
                os.environ['WANDB_API_KEY'] = args.wandb_token
                self.wandb = wandb
                if self.wandb.run is None:
                    self.wandb.init(project=f'fedscale-{args.job_name}',
                                    name=f'executor{args.this_rank}-{args.time_stamp}',
                                    group=f'{args.time_stamp}')
                else:
                    logging.error("Warning: wandb has already been initialized")
                
            else:
                self.wandb = None
        except Exception as e:
            logging.error(e)
            raise e
        super(Executor, self).__init__()

    def setup_env(self):
        """Set up experiments environment
        """
        logging.info(f"(EXECUTOR:{self.this_rank}) is setting up environ ...")
        self.setup_seed(seed=0)

    def setup_communication(self):
        """Set up grpc connection
        """
        self.init_control_communication()
        self.init_data_communication()

    def setup_seed(self, seed=1):
        """Set random seed for reproducibility

        Args:
            seed (int): random seed

        """
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def init_control_communication(self):
        """Create communication channel between coordinator and executor.
        This channel serves control messages.
        """
        self.aggregator_communicator.connect_to_server()

    def init_data_communication(self):
        """In charge of jumbo data traffics (e.g., fetch training result)
        """
        pass

    def init_data(self):
        """Return the training and testing dataset

        Returns:
            Tuple of DataPartitioner class: The partioned dataset class for training and testing

        """
        try:
            train_dataset, test_dataset = init_dataset()
            if self.args.task == "rl":
                return train_dataset, test_dataset
            if self.args.task == 'nlp':
                self.collate_fn = collate
            elif self.args.task == 'voice':
                self.collate_fn = voice_collate_fn
            # load data partitionxr (entire_train_data)
            logging.info("Data partitioner starts ...")

            # #Faraz - try to reset the seed to avoid random behaviour before every data loader
            # self.setup_seed(self.this_rank)
            
            # training_sets = DataPartitioner(
            #     data=train_dataset, args=self.args, numOfClass=self.args.num_class)
            
            # #Faraz - set the number of clients = total_clients if clientdatamap file is not present, or total_worker otherwise
            # if self.args.used_samples < 0:
            #     self.args.total_clients = int(len(train_dataset.data) / (self.args.batch_size+1))

            # num_clients = self.args.total_clients if self.args.total_clients > 0 else self.args.total_worker
            
            # if num_clients > 0:
            #     training_sets.partition_data_helper(
            #     num_clients=num_clients, data_map_file=self.args.data_map_file)
            # else:
                # training_sets.partition_data_helper(
                #     num_clients=self.args.num_participants, data_map_file=self.args.data_map_file)
            #Faraz - try to reset the seed to avoid random behaviour before every data loader
            self.setup_seed(seed=self.this_rank)
            num_clients = self.args.total_clients if self.args.total_clients > 0 else self.args.total_worker
            # logging.info(f"num_clients: {num_clients}")

            training_sets = DataPartitioner(data=train_dataset, args=self.args, numOfClass=self.args.num_class, seed=self.this_rank, isTest=False)
            #Faraz - set the number of clients = to total_clients if clientdatamap file is not present, or total_worker otherwise
            if self.args.used_samples < 0:
                self.args.total_clients = int(len(train_dataset.data) / (self.args.batch_size+1))

            training_sets.partition_data_helper(num_clients=num_clients, data_map_file=self.args.data_map_file)

            # Faraz - try to reset the seed to avoid random behaviour before every data loader
            self.setup_seed(seed=self.this_rank)

            testing_sets = DataPartitioner(data=test_dataset, args=self.args, numOfClass=self.args.num_class, seed=self.this_rank, isTest=True)
            testing_sets.partition_data_helper(num_clients=self.num_executors, data_map_file=self.args.data_map_file)
            
            validation_sets = DataPartitioner(data=train_dataset, args=self.args, numOfClass=self.args.num_class, seed=self.this_rank, isVal=True)
            validation_sets.partition_data_helper(num_clients=self.num_executors, data_map_file=self.args.data_map_file)
            
            # training_sets.partition_data_helper(
            #     num_clients=self.args.num_participants, data_map_file=self.args.data_map_file)
            # # Faraz - try to reset the seed to avoid random behaviour before every data loader
            # self.setup_seed(self.this_rank)
            
            # testing_sets = DataPartitioner(
            #     data=test_dataset, args=self.args, numOfClass=self.args.num_class, isTest=True)
            # testing_sets.partition_data_helper(num_clients=self.num_executors)

            logging.info("Data partitioner completes ...")

            return training_sets, validation_sets, testing_sets
        except Exception as e:
            logging.error(e)
            raise e

    def run(self):
        """Start running the executor by setting up execution and communication environment, and monitoring the grpc message.
        """
        try:
            self.setup_env()
            self.training_sets, self.validation_sets, self.testing_sets = self.init_data()
            self.setup_communication()
            self.event_monitor()
        except Exception as e:
            logging.error(e)
            raise e
        
    def dispatch_worker_events(self, request):
        """Add new events to worker queues

        Args:
            request (string): Add grpc request from server (e.g. MODEL_TEST, MODEL_TRAIN) to event_queue.

        """
        self.event_queue.append(request)

    def deserialize_response(self, responses):
        """Deserialize the response from server

        Args:
            responses (byte stream): Serialized response from server.

        Returns:
            ServerResponse defined at job_api.proto: The deserialized response object from server.

        """
        return pickle.loads(responses)

    def serialize_response(self, responses):
        """Serialize the response to send to server upon assigned job completion

        Args:
            responses (string, bool, or bytes): TorchClient responses after job completion.

        Returns:
            bytes stream: The serialized response object to server.

        """
        return pickle.dumps(responses)

    def UpdateModel(self, model_weights):
        """Receive the broadcasted global model for current round

        Args:
            config (PyTorch or TensorFlow model): The broadcasted global model config

        """
        try:
            self.round += 1
            self.model_adapter.set_weights(model_weights)
        except Exception as e:
            logging.error(e)
            raise e

    def compress_model(self, model, q_bit=16):
        try:
            # logging.info('HERE50')
            if type(model) is dict:
                model = [x for x in model.values()]
            # logging.info(f'Faraz debug - q_bit: {q_bit} and model type: {type(model)}')
            compressed_weights = []
            total_size = 0
            # logging.info('HERE53')
            for weight in model:
                # logging.info(f'Faraz debug - weight: {weight}')
                # logging.info('HERE54')
                compressed_weight, size = self.q_compressor.compress(weight, n_bit = q_bit)
                # logging.info('HERE55')
                total_size += int(size)
                # logging.info('HERE56')
                compressed_weights.append(compressed_weight)
                # logging.info(f'Faraz debug - compressed_weight: {compressed_weight}')
                # logging.info('HERE57')
                
            return compressed_weights, total_size
        except Exception as e:
            logging.error('Error in compressing model: ', e)
            raise e
    
    def prune_model(self, pruned_model, client_data, prune_percentage=0.25):
        try:
            pruned_model, reduction_ratio = self.pruning.prune_model(pruned_model, prune_percentage, client_data)
            return pruned_model, reduction_ratio
        
        except Exception as e:
            logging.error('Error in pruning model: ', e)
            raise e
        
    def decompress_model(self, model, q_bit=16):
        try:
            decompressed_weights = []
            for weight in model:
                decompressed_weight = self.q_compressor.decompress(weight, n_bit=q_bit)
                decompressed_weights.append(decompressed_weight)
            return decompressed_weights
        except Exception as e:
            logging.error('Error in decompressing model: ', e)
            raise e

    def Train(self, config):
        """Load train config and data to start training on that client

        Args:
            config (dictionary): The client training config.

        Returns:
            tuple (int, dictionary): The client id and train result

        """
        try:
            client_id, train_config, optimization = config['client_id'], config['task_config'], config['optimization'] if config.get('optimization') else None
            # logging.info(f"Received client train request for client {client_id} with optimization {optimization}")
            self.training_requests_received+=1
            if 'model' not in config or not config['model']:
                raise "The 'model' object must be a non-null value in the training config."
            client_conf = self.override_conf(train_config)
            try:
                train_res = self.training_handler(
                    client_id=client_id, conf=client_conf, model=config['model'], optimization=optimization)
            except Exception as e:
                logging.error(f"Error in training_handler: {e} of client {client_id}")
                raise e
            # logging.info(f"Training completed for client {client_id}")
            # Report execution completion meta information
            # logging.info(f'HERE60 for client: {client_id}')
            response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION(
                job_api_pb2.CompleteRequest(
                    client_id=str(client_id), executor_id=self.executor_id,
                    event=commons.CLIENT_TRAIN, status=True, msg=None,
                    meta_result=None, data_result=None
                )
            )
            # logging.info(f'HERE61 for client: {client_id}') 
            self.dispatch_worker_events(response)
            # logging.info(f'HERE62 for client: {client_id}') 
            return client_id, train_res
        except Exception as e:
            logging.error(f"Error in training: {e} of client {client_id}")
            raise e

    def Test(self, config):
        """Model Testing. By default, we test the accuracy on all data of clients in the test group

        Args:
            config (dictionary): The client testing config.

        """
        try:
            test_res = self.testing_handler()
            test_res = {'executorId': self.this_rank, 'results': test_res}

            # Report execution completion information
            response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION(
                job_api_pb2.CompleteRequest(
                    client_id=self.executor_id, executor_id=self.executor_id,
                    event=commons.MODEL_TEST, status=True, msg=None,
                    meta_result=None, data_result=self.serialize_response(test_res)
                )
            )
            self.dispatch_worker_events(response)
        except Exception as e:
            logging.error(e)
            raise e
    
    def log_validation_result(self, client_id, test_res, old_acc):
        try:
            """Faraz - log the validation result for each client"""
            acc = round(test_res["top_1"] / test_res["test_len"], 4)
            acc_5 = round(test_res["top_5"] / test_res["test_len"], 4)
            test_loss = test_res["test_loss"] / test_res["test_len"]
            # logging.info(f'log_validation_result old_acc: {old_acc}')
            if old_acc:
                self.old_client_losses[client_id] = test_loss
                self.old_client_accuracies[client_id] = acc
                logging.info(f"Client {client_id} - Old Validation Accuracy: {self.old_client_accuracies}")
            else:
                self.client_accuracies[client_id] = acc
                self.client_losses[client_id] = test_loss
        except Exception as e:
            logging.error(f'Error in logging validation result: {e}')
            raise e
        # logging.info(f"Client {client_id} - Validation Accuracy: {acc} - Top 5 Accuracy: {acc_5} - Loss: {test_loss}")
        
    def Val(self, config, round, clients, event):
        """Load train config and data to start training on that client

        Args:
            config (dictionary): The client training config.

        Returns:
            tuple (int, dictionary): The client id and train result

        """
        try:
            self.client_accuracies = {}
            self.client_losses = {}
            logging.info(f"Received client validation request for clients {clients} for round {round}")
            for client_id in clients:
                test_res = self.validation_handler(client_id, False)
                test_res = {'executorId': client_id, 'results': test_res}
                

            logging.info(f"Round: {round}")
            logging.info(f"Validation accuracies: {self.client_accuracies}")
            logging.info(f"Validation losses: {self.client_losses}")
            #Faraz - calculate the average accuracy, top 10% accuracy and bottom 10% accuracy
            if len(self.client_accuracies) > 9 and event == commons.CLIENT_VALIDATE_ALL:
                #calculate the average accuracy
                avg_acc = sum(self.client_accuracies.values()) / len(self.client_accuracies)
                #calculate top 10% accuracy
                sorted_acc = sorted(self.client_accuracies.values(), reverse=True)
                top_10_acc = sum(sorted_acc[:int(len(sorted_acc) * 0.1)]) / int(len(sorted_acc) * 0.1)
                #calculate bottom 10% accuracy
                bottom_10_acc = sum(sorted_acc[-int(len(sorted_acc) * 0.1):]) / int(len(sorted_acc) * 0.1)
                logging.info(f"Average accuracy: {avg_acc}, Top 10% accuracy: {top_10_acc}, Bottom 10% accuracy: {bottom_10_acc}")
            
            #get the difference between the old and new accuracies
            diff = {}
            logging.info(f"Old validation accuracies: {self.old_client_accuracies}")
            logging.info(f"New validation accuracies: {self.client_accuracies}")
            for client_id in self.client_accuracies:
                if client_id in self.old_client_accuracies:
                    diff[client_id] = self.client_accuracies[client_id] - self.old_client_accuracies[client_id]
                    # self.old_client_accuracies.remove(client_id)
                    # logging.info(f'Faraz -debug removed client {client_id} from old_client_accuracies')
                    # self.old_client_accuracies.pop(client_id)
                else:
                    logging.info(f"Old accuracy not found for client {client_id}")
                    diff[client_id] = self.client_accuracies[client_id]
            logging.info(f"Average accuracy difference: {diff}")
            # Report execution completion information
            if event == commons.CLIENT_VALIDATE:
                response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION(
                    job_api_pb2.CompleteRequest(
                        client_id=self.executor_id, executor_id=self.executor_id,
                        event=commons.CLIENT_VALIDATE, status=True, msg=None,
                        meta_result=None, data_result=self.serialize_response(diff)
                    )
                )
                self.dispatch_worker_events(response)
            else:
                response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION(
                    job_api_pb2.CompleteRequest(
                        client_id=self.executor_id, executor_id=self.executor_id,
                        event=commons.CLIENT_VALIDATE_ALL, status=True, msg=None,
                        meta_result=None, data_result=self.serialize_response(self.client_accuracies)
                    )
                )
                self.dispatch_worker_events(response)
            

        except Exception as e:
            logging.error(f"Error in Validation: {e}")
            raise e

    def Stop(self):
        """Stop the current executor
        """
        logging.info(f"Terminating the executor ...")
        self.aggregator_communicator.close_sever_connection()
        self.received_stop_request = True
        if self.wandb != None:
            self.wandb.finish()

    def report_executor_info_handler(self):
        """Return the statistics of training dataset

        Returns:
            int: Return the statistics of training dataset, in simulation return the number of clients

        """
        return self.training_sets.getSize()

    def override_conf(self, config):
        """ Override the variable arguments for different client

        Args:
            config (dictionary): The client runtime config.

        Returns:
            dictionary: Variable arguments for client runtime config.

        """
        default_conf = vars(self.args).copy()

        for key in config:
            default_conf[key] = config[key]

        return Namespace(**default_conf)

    def get_client_trainer(self, conf):
        """
        Returns a framework-specific client that handles training and evaluation.
        :param conf: job config
        :return: framework-specific client instance
        """
        try:
            if conf.engine == commons.TENSORFLOW:
                return TensorflowClient(conf)
            elif conf.engine == commons.PYTORCH:
                if conf.task == 'rl':
                    return RLClient(conf)
                else:
                    return TorchClient(conf)
            raise "Currently, FedScale supports tensorflow and pytorch."
        except Exception as e:
            raise e
        
    def get_quantized_client_trainer(self, conf):
        """
        Returns a framework-specific client that handles training and evaluation.
        :param conf: job config
        :return: framework-specific client instance
        """
        try:
            if conf.engine == commons.TENSORFLOW:
                return TensorflowClient(conf)
            elif conf.engine == commons.PYTORCH:
                if conf.task == 'rl':
                    return RLClient(conf)
                else:
                    return TorchClientQuantized(conf)
            raise "Currently, FedScale supports tensorflow and pytorch."
        except Exception as e:
            raise e

    def training_handler(self, client_id, conf, model, optimization=None):
        """Train model given client id

        Args:
            client_id (int): The client id.
            conf (dictionary): The client runtime config.

        Returns:
            dictionary: The train result

        """
        try:
            pruned_model = None
            q_bits = 16
            local_steps = None
            logging.info(f"Faraz - debug optimizaiton {optimization} in client {client_id}")
            if optimization and 'partial' in optimization:
                local_steps_reduction= 1.0 - int(optimization.split('_')[1])*0.01
                local_steps = int(local_steps_reduction*self.args.local_steps)
            if optimization and 'quantization' in optimization:
                q_bits = int(optimization.split('_')[1])
                # logging.info(f"Faraz - debug Before Training client {client_id} with model {model}")
                if optimization:
                    model = self.decompress_model(model, q_bit=q_bits)
                # logging.info(f"AFTER Faraz - debug optimizaiton {optimization} in client {client_id}")
            # logging.info(f"Faraz - debug After Training client {client_id} with model {model}")
            # for w in model:
            #     logging.info('model weights: {}'.format(w.shape))
            
            # if optimization != None:
            # logging.info('HERE45 for client {}'.format(client_id))
            # if optimization != None:
            #     logging.info('HERE47')
            conf.client_id = client_id
            conf.tokenizer = tokenizer
            client_data = self.training_sets if self.args.task == "rl" else \
                select_dataset(client_id, self.training_sets,
                            batch_size=conf.batch_size, args=self.args,
                            collate_fn=self.collate_fn
                            )
            
            # logging.info(f'HERE46 for client {client_id}')
            #Faraz - Handle pruning here after train_data is loaded
            if optimization and 'pruning' in optimization:
                prune_percentage = float(optimization.split('_')[1])*0.01
                self.model_adapter.set_weights(model)
                pruned_model, reduction_ratio = self.prune_model(self.model_adapter.get_model(), client_data, prune_percentage)
                # Get the weights as a list of torch tensors
                # weights_list = [weight.detach().cpu() for weight in pruned_model.parameters()]
                weights_list = [params.data.clone().cpu() for params in pruned_model.state_dict().values()]
                
                logging.info(f'Faraz - debug reduction_ratio: {reduction_ratio}')
                #compare shape of model and pruned model
                # for w_m, w_p in zip(model, weights_list):
                #     logging.info('model weights: {}'.format(w_m.shape))
                #     logging.info('pruned model weights: {}'.format(w_p.shape))
                self.model_adapter.set_weights(weights_list)
                
                # logging.info(f'HERE47 for client {client_id}')
            client = self.get_client_trainer(self.args)
            #Faraz - getting quantized aware trainer
            if optimization != None:
                client = self.get_client_trainer(self.args)
            # if optimization != None:
            # logging.info(f'HERE48 for client {client_id}')
            try:
                if optimization != None:
                    logging.info(f'Faraz - debug Validating client {client_id}')
                    test_res = self.validation_handler(client_id, True)
                train_res = client.train(
                    client_data=client_data, model=self.model_adapter.get_model(), conf=conf, local_steps=local_steps)
            except Exception as e:
                logging.error(f"Error in client.train {e}")
                return None
            # if optimization != None:
            # logging.info(f'HERE49 for client {client_id}')
            if optimization and 'quantization' in optimization:
                train_res['update_weight'], size = self.compress_model(train_res['update_weight'], q_bit=q_bits)
                train_res['optimization'] = optimization
                # logging.info('Compressed model: {}'.format(train_res['update_weight']))
            # logging.info(f'HERE51 for client {client_id}')
            return train_res
        except Exception as e:
            logging.error(f"Training error {e} in client {client_id}")
            return None

    def validation_handler(self, client_id, old_acc):
        """Test model

        Args:
            args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py
            config (dictionary): Variable arguments from coordinator.
        Returns:
            dictionary: The test result

        """
        try:
            # logging.info(f'validation_handler old_acc: {old_acc}')
            
            # logging.info(f"Total training requests received: {self.training_requests_received}")
            test_config = self.override_conf({
                'rank': self.this_rank,
                'memory_capacity': self.args.memory_capacity,
                'tokenizer': tokenizer
            })
            client = self.get_client_trainer(test_config)
            # logging.info(f"Faraz - debug validation_handler 626 {self.validation_sets}")
            data_loader = select_dataset(client_id, self.validation_sets,
                                        batch_size=self.args.test_bsz, args=self.args,
                                        isVal=True, collate_fn=self.collate_fn)
            # logging.info(f"Faraz - debug validation_handler 630 {data_loader}")
            # logging.info(f"Faraz - debug validation_handler 632 {data_loader.dataset}")
            test_results = client.test(data_loader, self.model_adapter.get_model(), test_config, isVal=True)
            self.log_validation_result(client_id, test_results, old_acc)
            
            gc.collect()

            return test_results

        except Exception as e:
            logging.error(f"Failed to validate the model {e}.")

    def testing_handler(self):
        """Test model

        Args:
            args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py
            config (dictionary): Variable arguments from coordinator.
        Returns:
            dictionary: The test result

        """
        try:
            # logging.info(f"Total training requests received: {self.training_requests_received}")
            self.training_requests_received = 0
            test_config = self.override_conf({
                'rank': self.this_rank,
                'memory_capacity': self.args.memory_capacity,
                'tokenizer': tokenizer
            })
            client = self.get_client_trainer(test_config)
            data_loader = select_dataset(self.this_rank, self.testing_sets,
                                        batch_size=self.args.test_bsz, args=self.args,
                                        isTest=True, collate_fn=self.collate_fn)

            test_results = client.test(data_loader, self.model_adapter.get_model(), test_config)
            self.log_test_result(test_results)
            gc.collect()

            return test_results
        except Exception as e:
            logging.error(f"Failed to test the model {e}.")

    def client_register(self):
        """Register the executor information to the aggregator
        """
        start_time = time.time()
        while time.time() - start_time < 180:
            try:
                response = self.aggregator_communicator.stub.CLIENT_REGISTER(
                    job_api_pb2.RegisterRequest(
                        client_id=self.executor_id,
                        executor_id=self.executor_id,
                        executor_info=self.serialize_response(
                            self.report_executor_info_handler())
                    )
                )
                self.dispatch_worker_events(response)
                break
            except Exception as e:
                logging.warning(f"Failed to connect to aggregator {e}. Will retry in 5 sec.")
                time.sleep(5)

    def client_ping(self):
        """Ping the aggregator for new task
        """
        try:
            response = self.aggregator_communicator.stub.CLIENT_PING(job_api_pb2.PingRequest(
                client_id=self.executor_id,
                executor_id=self.executor_id
            ))
            self.dispatch_worker_events(response)
        except Exception as e:
            logging.error(f"Failed to ping the aggregator {e}.")

    def event_monitor(self):
        """Activate event handler once receiving new message
        """
        logging.info("Start monitoring events ...")
        try:
            self.client_register()

            while not self.received_stop_request:
                if len(self.event_queue) > 0:
                    request = self.event_queue.popleft()
                    current_event = request.event

                    if current_event == commons.CLIENT_TRAIN:
                        train_config = self.deserialize_response(request.meta)
                        train_model = self.deserialize_response(request.data)
                        train_config['model'] = train_model
                        train_config['client_id'] = int(train_config['client_id'])
                        # if train_config['optimization']:
                        #     train_config['optimization'] = int(train_config['optimization'])
                        logging.info(f"Received client train request for client {train_config['client_id']}")
                        client_id, train_res = self.Train(train_config)
                        logging.info(f"Training completed for client {client_id}")
                        # Upload model updates
                        future_call = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION.future(
                            job_api_pb2.CompleteRequest(client_id=str(client_id), executor_id=self.executor_id,
                                                        event=commons.UPLOAD_MODEL, status=True, msg=None,
                                                        meta_result=None, data_result=self.serialize_response(train_res)
                                                        ))
                        future_call.add_done_callback(lambda _response: self.dispatch_worker_events(_response.result()))
                    
                    if current_event == commons.CLIENT_VALIDATE or current_event == commons.CLIENT_VALIDATE_ALL:
                        # logging.info("Received validation request meta from aggregator {} ...".format(self.deserialize_response(request.meta)))
                        # logging.info("Received validation request data from aggregator {} ...".format(self.deserialize_response(request.data)))
                        clients = self.deserialize_response(request.data)['clients']
                        round = self.deserialize_response(request.data)['round']
                        self.Val(self.deserialize_response(request.meta), round, clients, current_event)
                        
                    
                    elif current_event == commons.MODEL_TEST:
                        self.Test(self.deserialize_response(request.meta))

                    elif current_event == commons.UPDATE_MODEL:
                        model_weights = self.deserialize_response(request.data)
                        self.UpdateModel(model_weights)

                    elif current_event == commons.SHUT_DOWN:
                        self.Stop()

                    elif current_event == commons.DUMMY_EVENT:
                        pass
                else:
                    time.sleep(1)
                    try:
                        self.client_ping()
                    except Exception as e:
                        logging.info(f"Caught exception {e} from aggregator, terminating executor {self.this_rank} ...")
                        self.Stop()
        except Exception as e:
            logging.error(f"Caught exception {e} from executor {self.this_rank} ...")
            self.Stop()

    
    def log_test_result(self, test_res):
        """Log test results to wandb server if enabled
        """
        acc = round(test_res["top_1"] / test_res["test_len"], 4)
        acc_5 = round(test_res["top_5"] / test_res["test_len"], 4)
        test_loss = test_res["test_loss"] / test_res["test_len"]
        if self.wandb != None:
            self.wandb.log({
                'Test/round_to_top1_accuracy': acc,
                'Test/round_to_top5_accuracy': acc_5,
                'Test/round_to_loss': test_loss,
            }, step=self.round)

if __name__ == "__main__":
    executor = Executor(parser.args)
    executor.run()
