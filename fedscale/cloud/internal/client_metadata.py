import numpy as np
import random
import logging
from fedscale.dataloaders.divide_data import DataPartitioner
import os
import pickle


class ClientMetadata:
    """
    Contains the server-side metadata for a single client.
    """

    def __init__(self, host_id, client_id, speed, possible_bandwidths = [], traces=None):
        """
        Initializes the ClientMetadata.
        :param host_id: id of the executor which is handling this client.
        :param client_id: id of the client.
        :param speed: computation and communication speed of the client.
        :param traces: list of client availability traces.
        """
        random.seed(client_id)
        self.host_id = host_id
        self.client_id = client_id
        
        #Faraz - get samples per client from saved partitions
        # data_mappings = 'benchmark/dataset/data/femnist/metadata/femnist/data_mappings/part4_clients200_data637877_labels25_samples3189_alpha1.0'
        # data_mappings = 'benchmark/dataset/data/femnist/metadata/femnist/data_mappings/part4_clients200_data637877_labels25_samples3189_alpha0.05'
        # Access the environment variable
        # Construct the file path using the environment variable
        # data_mappings = 'benchmark/dataset/data/cifar/metadata/cifar10/data_mappings/part4_clients200_data50000_labels4_samples250_alpha0.1'
        # data_mappings = 'benchmark/dataset/data/cifar/metadata/cifar10/data_mappings/part-1_clients200_data50000_labels10_samples0'
        data_mappings = 'benchmark/dataset/data/femnist/metadata/femnist/data_mappings/part4_clients200_data637877_labels25_samples3189_alpha0.1'
        FLOAT_HOME = os.getcwd()
        FLOAT_HOME = os.path.join(FLOAT_HOME, 'FLOAT')
        filename = os.path.join(FLOAT_HOME, data_mappings)
        partitions = pickle.load(open(filename, 'rb'))
        # logging.info(f'partitions: {len(partitions)}')
        self.samples_per_client = len(partitions[client_id-1])
        self.compute_speed = speed['computation']
        self.compute_speed_with_inference = speed['computation']
        self.bandwidth = speed['communication']
        self.local_state = {'computation': self.compute_speed_with_inference, 'communication': self.bandwidth, 'deadline_difference': 0}
        #Faraz - get 1 random value from possible_bandwidths
        if len(possible_bandwidths)<=0:
            self.possible_bandwidths = [self.bandwidth]
        else:
            self.possible_bandwidths = possible_bandwidths
        self.bandwidth = random.sample(self.possible_bandwidths, 1)
        if type(self.bandwidth) is list:
            self.bandwidth = self.bandwidth[0]
        logging.info("Client {} has bandwidth {}".format(self.client_id, self.bandwidth))
        self.score = 0
        #Faraz - add client participation rate
        self.participation_rate = 0
        self.traces = traces
        self.behavior_index = 0
        #Faraz - current completion time of the client
        self.completion_time = {'computation': 0, 'communication': 0}

    def get_score(self):
        return self.score

    def register_reward(self, reward):
        """
        Registers the rewards of the client
        :param reward: int
        """
        self.score = reward

    #Faraz - get client resources
    def get_client_resources(self):
        """
        Returns the resources of the client
        :return: dict
        """
        return {'computation_power': self.compute_speed_with_inference, 'communication_kbps': self.bandwidth}
    
    def is_active(self, cur_time):
        """
        Decides whether the client is active at given cur_time
        :param cur_time: time in seconds
        :return: boolean
        """
        if self.traces is None:
            return True

        norm_time = cur_time % self.traces['finish_time']

        if norm_time > self.traces['inactive'][self.behavior_index]:
            self.behavior_index += 1
            # logging.info("Client {} behavior index increasing {}".format(self.client_id, self.behavior_index))

        self.behavior_index %= len(self.traces['active'])
        if self.traces['active'][self.behavior_index] <= norm_time <= self.traces['inactive'][self.behavior_index]:
            # logging.info("Client {} is active: {} - norm: {} - inactive: {} - curr_time: {} - finish_time {}, behavior_index: {}".format(self.client_id, self.traces['active'][self.behavior_index], norm_time, self.traces['inactive'][self.behavior_index], cur_time, self.traces['finish_time'], self.behavior_index))
            return True

        return False
    
    def is_activewithDeadline(self, cur_time):
        """
        Decides whether the client is active at given cur_time
        :param cur_time: time in seconds
        :return: boolean
        """
        if self.traces is None:
            return True, 0

        norm_time = cur_time % self.traces['finish_time']

        # if norm_time > self.traces['inactive'][self.behavior_index]:
        #     self.behavior_index += 1

        self.behavior_index %= len(self.traces['active'])

        if self.traces['active'][self.behavior_index] <= norm_time <= self.traces['inactive'][self.behavior_index]:
            return True, 0

        self.local_state = {'computation': self.compute_speed_with_inference, 'communication': self.bandwidth, 'deadline_difference': self.traces['inactive'][self.behavior_index] - norm_time}
        logging.info("Client {} is inactive: {} - norm: {} - inactive: {} - curr_time: {} - finish_time {}, behavior_index: {}".format(self.client_id, self.traces['active'][self.behavior_index], norm_time, self.traces['inactive'][self.behavior_index], cur_time, self.traces['finish_time'], self.behavior_index))
        return False, self.traces['inactive'][self.behavior_index] - norm_time
    
    def can_participate(self, curr_time, compute_time, communication_time):
        """
        Decides whether the client is active at given total_time
        :param total_time: time in seconds
        :return: boolean
        """
        try:
            total_time = communication_time + compute_time
            if self.traces is None:
                return True, 0
            
            # Faraz - reset behavior index if the client is reaching finish time of trace
            norm_time = curr_time % self.traces['finish_time']

            if norm_time > self.traces['inactive'][self.behavior_index]:
                self.behavior_index += 1
            
            self.behavior_index %= len(self.traces['active'])
            #Faraz - calculate end time based on whether request is recieved at start of active time or midway
            end_time = max(norm_time, self.traces['active'][self.behavior_index]) + total_time
            
            self.local_state = {'computation': self.compute_speed_with_inference, 'communication': self.bandwidth, 'deadline_difference': self.traces['inactive'][self.behavior_index] - end_time}
            #Faraz - check which bottleneck is causing the client to be inactive
            if  end_time <= self.traces['inactive'][self.behavior_index]:
                return (True,  0)
            else:
                if max(norm_time, self.traces['active'][self.behavior_index]) + compute_time > self.traces['inactive'][self.behavior_index]:
                    logging.info("Client {} has compute bottleneck: {} ".format(self.client_id, self.traces['inactive'][self.behavior_index] - (self.traces['active'][self.behavior_index] + compute_time)))
                    # logging.info("Client {} is not active: {} - norm: {} - inactive: {} - curr_time: {} - finish_time {}, behavior_index: {}, total_time: {}, end_time: {}".format(self.client_id, self.traces['active'][self.behavior_index], norm_time, self.traces['inactive'][self.behavior_index], curr_time, self.traces['finish_time'], self.behavior_index, total_time, end_time))
                    return (False, self.traces['inactive'][self.behavior_index] - end_time)
                elif max(norm_time, self.traces['active'][self.behavior_index]) + communication_time > self.traces['inactive'][self.behavior_index]:
                    logging.info("Client {} has communication bottleneck: {} ".format(self.client_id, self.traces['inactive'][self.behavior_index] - (self.traces['active'][self.behavior_index] + communication_time)))
                    # logging.info("Client {} is not active: {} - norm: {} - inactive: {} - curr_time: {} - finish_time {}, behavior_index: {}, total_time: {}, end_time: {}".format(self.client_id, self.traces['active'][self.behavior_index], norm_time, self.traces['inactive'][self.behavior_index], curr_time, self.traces['finish_time'], self.behavior_index, total_time, end_time))
                    return (False, self.traces['inactive'][self.behavior_index] - end_time)
                elif end_time > self.traces['inactive'][self.behavior_index]:
                    logging.info("Client {} has both bottlenecks: {} ".format(self.client_id, self.traces['inactive'][self.behavior_index] - end_time))
                    # logging.info("Client {} is not active: {} - norm: {} - inactive: {} - curr_time: {} - finish_time {}, behavior_index: {}, total_time: {}, end_time: {}".format(self.client_id, self.traces['active'][self.behavior_index], norm_time, self.traces['inactive'][self.behavior_index], curr_time, self.traces['finish_time'], self.behavior_index, total_time, end_time))
                    return (False, self.traces['inactive'][self.behavior_index] - end_time)
        except Exception as e:
            logging.info("Exception in can_participate: {}".format(e))
            return (True, 0)

    def get_new_network_bandwidth(self):
        """
        Returns the new network bandwidth of the client
        :return: float
        """
        self.bandwidth = random.sample(self.possible_bandwidths, 1)
        if type(self.bandwidth) is list:
            self.bandwidth = self.bandwidth[0]
        return self.bandwidth
    
    
    def get_completion_time_with_variable_network(self, batch_size, local_steps, upload_size, download_size, augmentation_factor=3.0, add_cpu_noise=True, add_network_noise=True):
        """
           Computation latency: compute_speed is the inference latency of models (ms/sample). As reproted in many papers,
                                backward-pass takes around 2x the latency, so we multiple it by 3x;
           Communication latency: communication latency = (pull + push)_update_size/bandwidth;
        """
        try:
            self.bandwidth = self.get_new_network_bandwidth()
            if add_cpu_noise:
                self.compute_speed_with_inference = np.random.uniform(0.1*self.compute_speed, 1.0*self.compute_speed)
                # self.compute_speed_with_inference = np.random.uniform(0.01*self.compute_speed, 0.1*self.compute_speed)
                # self.compute_speed_with_inference = 2*self.compute_speed
                #high_network_low_cpu
                # self.compute_speed_with_inference = self.compute_speed
            if add_network_noise:
                # self.bandwidth = np.random.uniform(0.0001*self.bandwidth, 0.001*self.bandwidth)
                self.bandwidth = np.random.uniform(0.1*self.bandwidth, 1.0*self.bandwidth)
                # self.bandwidth = 0.2*self.bandwidth
            if self.bandwidth == 0 or self.compute_speed_with_inference == 0:
                self.completion_time = {'computation': 10000, 'communication': 10000}
                return self.completion_time
            
            # logging.info(f'Faraz - debug augmentation_factor: {augmentation_factor}, batch_size: {self.samples_per_client}, local_steps: {local_steps}, compute_speed_with_inference: {self.compute_speed_with_inference/1000}, bandwidth: {self.bandwidth}')
            # self.completion_time = {'computation': augmentation_factor * batch_size * local_steps*float(self.compute_speed_with_inference)/1000.,
            #         'communication': (upload_size+download_size)/float(self.bandwidth)}
            self.completion_time = {'computation': augmentation_factor * self.samples_per_client * local_steps*float(self.compute_speed_with_inference)/1000.,
                    'communication': (upload_size+download_size)/float(self.bandwidth)}
            
            return self.completion_time
        except Exception as e:
            logging.info("Error in get_completion_time_with_variable_network: {}".format(e))
            
        
    def get_completion_time(self, batch_size, local_steps, upload_size, download_size, augmentation_factor=3.0, variable_resources = False):
        """
           Computation latency: compute_speed is the inference latency of models (ms/sample). As reproted in many papers,
                                backward-pass takes around 2x the latency, so we multiple it by 3x;
           Communication latency: communication latency = (pull + push)_update_size/bandwidth;
        """
        if variable_resources:
            self.completion_time = {'computation': augmentation_factor * batch_size * local_steps*float(self.compute_speed_with_inference)/1000.,
                'communication': (upload_size+download_size)/float(self.bandwidth)}
        else:
            self.completion_time = {'computation': augmentation_factor * batch_size * local_steps*float(self.compute_speed)/1000.,
                'communication': (upload_size+download_size)/float(self.bandwidth)}
        return self.completion_time

    def get_completion_time_lognormal(self, batch_size, local_steps, upload_size, download_size,
                                      mean_seconds_per_sample=0.005, tail_skew=0.6):
        """
        Computation latency: compute_speed is the inference latency of models (ms/sample). The calculation assumes
        that client computation speed is a lognormal distribution (see PAPAPYA / GFL papers), and uses the parameters
        to sample a client task completion time.
        Communication latency: communication latency = (pull + push)_update_size/bandwidth;
        :param batch_size: size of each training batch
        :param local_steps: number of local client training steps
        :param upload_size: size of model download (MB)
        :param download_size: size of model upload (MB)
        :param mean_seconds_per_sample: mean seconds to process a single training example. This can be adjusted based on
        on-device benchmarks.
        :param tail_skew: the skew of the lognormal distribution used to model device speed.
        :return: dict of computation and communication times for the client's training task.
        """
        device_speed = max(0.0001, np.random.lognormal(1, tail_skew, 1)[0])
        return {'computation': device_speed * mean_seconds_per_sample * batch_size * local_steps,
                'communication': (upload_size + download_size) / float(self.bandwidth)}
        
    #Faraz - added this function to get the quota for the client
    
    def get_quota(self):
        """
           Computation latency: compute_speed is the inference latency of models (ms/sample). As reproted in many papers,
                                backward-pass takes around 2x the latency, so we multiple it by 3x;
           Communication latency: communication latency = (pull + push)_update_size/bandwidth;
        """
        if self.traces is None or 'quota' not in self.traces:
            return {'quota': 10000000}
        else:
            return {'quota': self.traces['quota']}
