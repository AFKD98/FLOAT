import logging
import numpy as np
import random
import pickle

class Client(object):

    def __init__(self, hostId, clientId, speed, possible_bandwidths, traces=None):
        random.seed(clientId)
        self.hostId = hostId
        self.clientId = clientId
        #Ahmed - based on the format from the device trace file key:432802 val:{'computation': 162.0, 'communication': 5648.109619499134}
        self.compute_speed = speed['computation']
        self.bandwidth = speed['communication']
        self.compute_speed_with_inference = speed['computation']
        self.bandwidth = speed['communication']
        #Faraz - get 1 random value from possible_bandwidths
        if len(possible_bandwidths)<=0:
            self.possible_bandwidths = [self.bandwidth]
        else:
            self.possible_bandwidths = possible_bandwidths
        self.bandwidth = random.sample(self.possible_bandwidths, 1)
        if type(self.bandwidth) is list:
            self.bandwidth = self.bandwidth[0]
        logging.info("Client {} has bandwidth {}".format(self.clientId, self.bandwidth))
        self.score = 0
        self.traces = traces
        self.behavior_index = 0
        # filename = '/home/ahmad/FedScale/benchmark/dataset/data/femnist/metadata/femnist/data_mappings/part4_clients200_data637877_labels25_samples3189_alpha0.1'
        # filename = '/home/ahmad/REFL/dataset/data/femnist/metadata/femnist/data_mappings/part4_clients200_data637877_labels25_samples3189_alpha0.1'
        filename = '/home/ahmad/REFL/dataset/data/cifar10/metadata/cifar10/data_mappings/part4_clients200_data50000_labels4_samples250_alpha0.1'
        partitions = pickle.load(open(filename, 'rb'))
        logging.info(f'len partitions: {len(partitions[clientId-1])}')
        # logging.info(f'partitions: {len(partitions)}')
        self.samples_per_client = len(partitions[clientId-1])
        
    def getScore(self):
        return self.score

    def registerReward(self, reward):
        self.score = reward

    #TODO: clarify this part on the use of the trace!
    #Ahmed - the trace pickle file contains only 107,749 clients!
    #Format- key:3834 val:{'duration': 211625, 'inactive': [65881, 133574, 208292, 276575, 295006, 356236, 400906, 475099], 'finish_time': 518400, 'active': [12788, 100044, 188992, 271372, 276663, 352625, 356267, 441193], 'model': 'CPH1801'}
    def isActive(self, cur_time):
        if self.traces is None:
            return True

        norm_time = cur_time % self.traces['finish_time']

        if norm_time > self.traces['inactive'][self.behavior_index]:
            self.behavior_index += 1

        self.behavior_index %= len(self.traces['active'])

        if (self.traces['active'][self.behavior_index] <= norm_time <= self.traces['inactive'][self.behavior_index]):
            return True
        return False

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
            
            #Faraz - check which bottleneck is causing the client to be inactive
            if  end_time <= self.traces['inactive'][self.behavior_index]:
                return (True,  0)
            else:
                if max(norm_time, self.traces['active'][self.behavior_index]) + compute_time > self.traces['inactive'][self.behavior_index]:
                    logging.info("Client {} has compute bottleneck: {} ".format(self.clientId, self.traces['inactive'][self.behavior_index] - (self.traces['active'][self.behavior_index] + compute_time)))
                    # logging.info("Client {} is not active: {} - norm: {} - inactive: {} - curr_time: {} - finish_time {}, behavior_index: {}, total_time: {}, end_time: {}".format(self.clientId, self.traces['active'][self.behavior_index], norm_time, self.traces['inactive'][self.behavior_index], curr_time, self.traces['finish_time'], self.behavior_index, total_time, end_time))
                    return (False, self.traces['inactive'][self.behavior_index] - end_time)
                elif max(norm_time, self.traces['active'][self.behavior_index]) + communication_time > self.traces['inactive'][self.behavior_index]:
                    logging.info("Client {} has communication bottleneck: {} ".format(self.clientId, self.traces['inactive'][self.behavior_index] - (self.traces['active'][self.behavior_index] + communication_time)))
                    # logging.info("Client {} is not active: {} - norm: {} - inactive: {} - curr_time: {} - finish_time {}, behavior_index: {}, total_time: {}, end_time: {}".format(self.clientId, self.traces['active'][self.behavior_index], norm_time, self.traces['inactive'][self.behavior_index], curr_time, self.traces['finish_time'], self.behavior_index, total_time, end_time))
                    return (False, self.traces['inactive'][self.behavior_index] - end_time)
                elif end_time > self.traces['inactive'][self.behavior_index]:
                    logging.info("Client {} has both bottlenecks: {} ".format(self.clientId, self.traces['inactive'][self.behavior_index] - end_time))
                    # logging.info("Client {} is not active: {} - norm: {} - inactive: {} - curr_time: {} - finish_time {}, behavior_index: {}, total_time: {}, end_time: {}".format(self.clientId, self.traces['active'][self.behavior_index], norm_time, self.traces['inactive'][self.behavior_index], curr_time, self.traces['finish_time'], self.behavior_index, total_time, end_time))
                    return (False, self.traces['inactive'][self.behavior_index] - end_time)
        except Exception as e:
            logging.info("Exception in can_participate: {}".format(e))
            return (True, 0)
    #Ahmed - return the availability windows of the client
    def availabilityPeriods(self):
        period_list=[]
        for i in range(len(self.traces['inactive'])):
            period_list.append((self.traces['active'][i], self.traces['inactive'][i]))
        return period_list

    #TODO clarify the contents of the device compute trace
    #Ahmed - the trace pickle file contains only 500,000 clients!
    #Format - key:432802 val:{'computation': 162.0, 'communication': 5648.109619499134}
    
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
                # self.compute_speed_with_inference = np.random.uniform(0.1*self.compute_speed, self.compute_speed)
                self.compute_speed_with_inference = 2*self.compute_speed
            if add_network_noise:
                # self.bandwidth = np.random.uniform(0.1*self.bandwidth, 1.0*self.bandwidth)
                self.bandwidth = 0.2*self.bandwidth
            if self.bandwidth == 0 or self.compute_speed_with_inference == 0:
                self.completion_time = {'computation': 10000000, 'communication': 10000000}
                return self.completion_time


            # self.completion_time = {'computation': augmentation_factor * batch_size * local_steps*float(self.compute_speed_with_inference)/1000.,
            #         'communication': (upload_size+download_size)/float(self.bandwidth)}
            # logging.info("139: Client {} has completion time: {}".format(self.clientId, self.completion_time))
            self.completion_time = {'computation': augmentation_factor * self.samples_per_client * local_steps*float(self.compute_speed_with_inference)/1000.,
                    'communication': (upload_size+download_size)/float(self.bandwidth)}
            # logging.info("142: Client {} has completion time: {}".format(self.clientId, self.completion_time))
            return self.completion_time
        except Exception as e:
            logging.info("Error in get_completion_time_with_variable_network: {}".format(e))
            
    def getCompletionTime(self, batch_size, upload_epoch, upload_size, download_size, augmentation_factor=3.0):
        """
           Computation latency: compute_speed is the inference latency of models (ms/sample). As reproted in many papers, 
                                backward-pass takes around 2x the latency, so we multiple it by 3x;
           Communication latency: communication latency = (pull + push)_update_size/bandwidth;
        """
        return {'computation':augmentation_factor * batch_size * upload_epoch*float(self.compute_speed)/1000., \
                'communication': (upload_size+download_size)/float(self.bandwidth)}
