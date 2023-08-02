import logging
import math
import pickle
from random import Random
from typing import Dict, List
import random
from fedscale.cloud.internal.client_metadata import ClientMetadata
import numpy as np

class ClientManager:

    def __init__(self, mode, args, sample_seed=233):
        self.client_metadata = {}
        self.client_on_hosts = {}
        self.mode = args.mode
        logging.info("Client manager mode: {}".format(self.mode))
        self.filter_less = args.filter_less
        self.filter_more = args.filter_more

        self.ucb_sampler = None

        if self.mode == 'oort':
            from thirdparty.oort.oort import create_training_selector
            self.ucb_sampler = create_training_selector(args=args)

        self.feasibleClients = []
        self.rng = Random()
        # Set the random seed for reproducibility
        random.seed(sample_seed)
        # self.rng.seed(sample_seed)
        self.count = 0
        self.feasible_samples = 0
        self.user_trace = None
        self.args = args
        
        if args.device_avail_file is not None:
            with open(args.device_avail_file, 'rb') as fin:
                self.user_trace = pickle.load(fin)
            self.user_trace_keys = list(self.user_trace.keys())

    #Faraz - get client behavior index
    def get_client_behavior_index(self, client_id):
        return self.client_metadata[self.getUniqueId(0, client_id)].behavior_index
    
    def get_client_local_state(self, client_id):
        local_state = self.client_metadata[self.getUniqueId(0, client_id)].local_state
        compute = local_state['computation']
        bandwidth = local_state['communication']
        deadline_difference = local_state['deadline_difference']
        translated_local_state = {}
        logging.info('Faraz - bandwidth: {}, compute: {}, deadline_difference: {}'.format(bandwidth, compute, deadline_difference))
        # if bandwidth < 53760:
        #     translated_local_state['communication'] = 'low'
        # elif bandwidth < 85913:
        #     translated_local_state['communication'] = 'medium'
        # else:
        #     translated_local_state['communication'] = 'high'
        
        #get all compute values
        resource_values = self.getAllClientsResources()
        compute_values = [x['computation_power'] for x in resource_values]
        communication_values = [x['communication_kbps'] for x in resource_values]
        max_compute = max(compute_values)
        q1, q2, q3, q4 = 0, 0, 0, 0
        #fit translated_local_states into buckets for computation and communication according to the percentiles
        resources_to_categorize = ['computation', 'communication']
        for resource in resources_to_categorize:
            if resource == 'computation':
                resource_var = compute
                q1, q2, q3, q4 = np.percentile(compute_values, [10, 20, 30, 40])
            else:
                resource_var = bandwidth
                q1, q2, q3, q4 = np.percentile(communication_values, [10, 20, 30, 40])
            #see which bucket the compute value falls into from all percentiles
            if resource_var < q1:
                translated_local_state[resource] = '1'
            elif resource_var < q2:
                translated_local_state[resource] = '2'
            elif resource_var < q3:
                translated_local_state[resource] = '3'
            elif resource_var < q4:
                translated_local_state[resource] = '4'
            else:
                translated_local_state[resource] = '5'
                    
        # if compute < q1:
        #     translated_local_state['computation'] = 'low'
        # elif compute < q2:
        #     translated_local_state['computation'] = 'medium'
        # else:
        #     translated_local_state['computation'] = 'high'
        
        if deadline_difference > -2:
            translated_local_state['deadline_difference'] = '1'
        elif deadline_difference > -10:
            translated_local_state['deadline_difference'] = '2'
        elif deadline_difference > -20:
            translated_local_state['deadline_difference'] = '3'
        elif deadline_difference > -30:
            translated_local_state['deadline_difference'] = '4'
        else:
            translated_local_state['deadline_difference'] = '5'
                    
        return translated_local_state
            
    
    def getAllClientsResources(self):
        ''' Returns a list of all clients' resources '''
        return [self.client_metadata[self.getUniqueId(0, client_id)].get_client_resources() for client_id in self.feasibleClients]
    
    def get_client_local_states(self, client_ids):
        return [self.client_metadata[self.getUniqueId(0, client_id)].local_state for client_id in client_ids]
    
    def register_client(self, host_id: int, client_id: int, size: int, speed: Dict[str, float], possible_bandwidths: List,
                        duration: float = 1) -> None:
        """Register client information to the client manager.

        Args:
            host_id (int): executor Id.
            client_id (int): client Id.
            size (int): number of samples on this client.
            speed (Dict[str, float]): device speed (e.g., compuutation and communication).
            duration (float): execution latency.

        """
        uniqueId = self.getUniqueId(host_id, client_id)
        user_trace = None if self.user_trace is None else self.user_trace[self.user_trace_keys[int(
            client_id) % len(self.user_trace)]]

        self.client_metadata[uniqueId] = ClientMetadata(host_id, client_id, speed, possible_bandwidths, user_trace)
        # remove clients
        if size >= self.filter_less and size <= self.filter_more:
            self.feasibleClients.append(client_id)
            self.feasible_samples += size

            if self.mode == "oort":
                feedbacks = {'reward': min(size, self.args.local_steps * self.args.batch_size),
                             'duration': duration,
                             }
                self.ucb_sampler.register_client(client_id, feedbacks=feedbacks)
        else:
            del self.client_metadata[uniqueId]

    def get_client_resources(self, client_id):
        return self.client_metadata[self.getUniqueId(0, client_id)].get_client_resources()
        
    def getAllClients(self):
        return self.feasibleClients

    def getAllClientsLength(self):
        return len(self.feasibleClients)

    def getClient(self, client_id):
        return self.client_metadata[self.getUniqueId(0, client_id)]

    def registerDuration(self, client_id, batch_size, local_steps, upload_size, download_size):
        if self.mode == "oort" and self.getUniqueId(0, client_id) in self.client_metadata:
            exe_cost = self.client_metadata[self.getUniqueId(0, client_id)].get_completion_time(
                batch_size=batch_size, local_steps=local_steps,
                upload_size=upload_size, download_size=download_size
            )
            self.ucb_sampler.update_duration(
                client_id, exe_cost['computation'] + exe_cost['communication'])

    def get_completion_time(self, client_id, batch_size, local_steps, upload_size, download_size, variable_resources=False):
        return self.client_metadata[self.getUniqueId(0, client_id)].get_completion_time(
            batch_size=batch_size, local_steps=local_steps,
            upload_size=upload_size, download_size=download_size, variable_resources=variable_resources
        )
    
    def get_completion_time_with_variable_network(self, client_id, batch_size, local_steps, upload_size, download_size):
        try:
            return self.client_metadata[self.getUniqueId(0, client_id)].get_completion_time_with_variable_network(
                batch_size=batch_size, local_steps=local_steps,
                upload_size=upload_size, download_size=download_size
            )
        except Exception as e:
            logging.error("Error in get_completion_time_with_variable_network: {}".format(e))
        
    def generate_random_float(self, skewness=0.5):
        # Generate a random number from the exponential distribution
        # Adjust the scale parameter to control the distribution shape
        value = random.expovariate(skewness)
        return value    
    
    def register_possible_speeds(self, client_id, possible_bandwidths):
        self.client_metadata[self.getUniqueId(0, client_id)].possible_bandwidths = possible_bandwidths
        
    def registerSpeed(self, host_id, client_id, speed):
        uniqueId = self.getUniqueId(host_id, client_id)
        self.client_metadata[uniqueId].speed = speed

    def registerScore(self, client_id, reward, auxi=1.0, time_stamp=0, duration=1., success=True):
        self.register_feedback(client_id, reward, auxi=auxi, time_stamp=time_stamp, duration=duration, success=success)

    def register_feedback(self, client_id: int, reward: float, auxi: float = 1.0, time_stamp: float = 0,
                          duration: float = 1., success: bool = True) -> None:
        """Collect client execution feedbacks of last round.

        Args:
            client_id (int): client Id.
            reward (float): execution utilities (processed feedbacks).
            auxi (float): unprocessed feedbacks.
            time_stamp (float): current wall clock time.
            duration (float): system execution duration.
            success (bool): whether this client runs successfully.

        """
        # currently, we only use distance as reward
        if self.mode == "oort":
            feedbacks = {
                'reward': reward,
                'duration': duration,
                'status': True,
                'time_stamp': time_stamp
            }

            self.ucb_sampler.update_client_util(client_id, feedbacks=feedbacks)

    def registerClientScore(self, client_id, reward):
        self.client_metadata[self.getUniqueId(0, client_id)].register_reward(reward)

    def get_score(self, host_id, client_id):
        uniqueId = self.getUniqueId(host_id, client_id)
        return self.client_metadata[uniqueId].get_score()

    def getClientsInfo(self):
        clientInfo = {}
        for i, client_id in enumerate(self.client_metadata.keys()):
            client = self.client_metadata[client_id]
            clientInfo[client.client_id] = client.distance
        return clientInfo

    def next_client_id_to_run(self, host_id):
        init_id = host_id - 1
        lenPossible = len(self.feasibleClients)

        while True:
            client_id = str(self.feasibleClients[init_id])
            csize = self.client_metadata[client_id].size
            if csize >= self.filter_less and csize <= self.filter_more:
                return int(client_id)

            init_id = max(
                0, min(int(math.floor(self.rng.random() * lenPossible)), lenPossible - 1))

    def getUniqueId(self, host_id, client_id):
        return str(client_id)
        # return (str(host_id) + '_' + str(client_id))

    def clientSampler(self, client_id):
        return self.client_metadata[self.getUniqueId(0, client_id)].size

    def clientOnHost(self, client_ids, host_id):
        self.client_on_hosts[host_id] = client_ids

    def getCurrentclient_ids(self, host_id):
        return self.client_on_hosts[host_id]

    def getClientLenOnHost(self, host_id):
        return len(self.client_on_hosts[host_id])

    def getClientSize(self, client_id):
        return self.client_metadata[self.getUniqueId(0, client_id)].size

    def getSampleRatio(self, client_id, host_id, even=False):
        totalSampleInTraining = 0.

        if not even:
            for key in self.client_on_hosts.keys():
                for client in self.client_on_hosts[key]:
                    uniqueId = self.getUniqueId(key, client)
                    totalSampleInTraining += self.client_metadata[uniqueId].size

            # 1./len(self.client_on_hosts.keys())
            return float(self.client_metadata[self.getUniqueId(host_id, client_id)].size) / float(totalSampleInTraining)
        else:
            for key in self.client_on_hosts.keys():
                totalSampleInTraining += len(self.client_on_hosts[key])

            return 1. / totalSampleInTraining

    def getFeasibleClients(self, cur_time):
        if self.user_trace is None:
            clients_online = self.feasibleClients
        else:
            # logging.info(f'Faraz - Getting feasible clients at time {round(cur_time)}')
            clients_online = [client_id for client_id in self.feasibleClients if self.client_metadata[self.getUniqueId(
                0, client_id)].is_active(cur_time)]
            
        logging.info(f"Wall clock time: {round(cur_time)}, {len(clients_online)} clients online, " +
                     f"{len(self.feasibleClients) - len(clients_online)} clients offline")

        return clients_online
    
    def getClientFeasibilityForParticipation(self, client_id, cur_time, upload_size = 0, download_size = 0, random_client_quota = False, deadline = 0):
        try:
            if self.user_trace is None:
                return (True, 0)
            else:
                #Faraz - 1. Check if client is active
                if self.client_metadata[self.getUniqueId(0, client_id)].is_active(cur_time):
                    #Faraz - 2. Check if client will be able to meet the deadline
                    compute_time = self.client_metadata[self.getUniqueId(0, client_id)].completion_time['computation']
                    communication_time = self.client_metadata[self.getUniqueId(0, client_id)].completion_time['communication']
                    #Faraz - 3. Check if client can deliver within the deadline and before it becomes inactive
                    (will_participate, deadline_difference) = self.client_metadata[self.getUniqueId(0, client_id)].can_participate(cur_time, compute_time, communication_time)
                    if deadline > 0 and will_participate:
                        if compute_time+communication_time < deadline and will_participate:
                            return (True, deadline_difference)
                    elif deadline == 0 and will_participate:
                        return (will_participate, deadline_difference)
                    else:
                        return (False, deadline_difference)
                else:
                    logging.info(f"Faraz - Client {client_id} is not active at time {cur_time}")
                    return (False, 0)
        except Exception as e:
            logging.info(f"Faraz - Exception in getClientFeasibilityForParticipation: {e}")
            return (False, 0)

    def isClientActivewithDeadline(self, client_id, cur_time):
        return self.client_metadata[self.getUniqueId(0, client_id)].is_activewithDeadline(cur_time)
    
    def isClientActive(self, client_id, cur_time):
        return self.client_metadata[self.getUniqueId(0, client_id)].is_active(cur_time)

    def select_participants(self, num_of_clients: int, cur_time: float = 0) -> List[int]:
        """Select participating clients for current execution task.

        Args:
            num_of_clients (int): number of participants to select.
            cur_time (float): current wall clock time.

        Returns:
            List[int]: indices of selected clients.

        """
        try:
            self.count += 1

            clients_online = self.getFeasibleClients(cur_time)
            # if self.mode == "fedAvg":
            #     clients_online = self.getAllClients()
            if len(clients_online) <= num_of_clients:
                return clients_online

            pickled_clients = None
            clients_online_set = set(clients_online)

            if self.mode == "oort" and self.count > 1:
                pickled_clients = self.ucb_sampler.select_participant(
                    num_of_clients, feasible_clients=clients_online_set)
            elif self.mode == "FLOAT":
                self.rng.shuffle(clients_online)
                client_len = min(num_of_clients, len(clients_online) - 1)
                #Faraz - temporarily changing it for RL agent
                pickled_clients = clients_online[:client_len]
            elif self.mode == "fedAvg":
                self.rng.shuffle(clients_online)
                client_len = min(num_of_clients, len(clients_online) - 1)
                #Faraz - temporarily changing it for RL agent
                pickled_clients = clients_online[:client_len]
                # pickled_clients = random.sample(clients_online, client_len)
            else:
                self.rng.shuffle(clients_online)
                client_len = min(num_of_clients, len(clients_online) - 1)
                pickled_clients = random.sample(clients_online, client_len)
                # pickled_clients = clients_online[:client_len]

            return pickled_clients
        except Exception as e:
            logging.error(f"Error in selecting participants: {e}")
            return []

    def resampleClients(self, num_of_clients, cur_time=0):
        return self.select_participants(num_of_clients, cur_time)

    def getAllMetrics(self):
        if self.mode == "oort":
            return self.ucb_sampler.getAllMetrics()
        return {}

    def getDataInfo(self):
        return {'total_feasible_clients': len(self.feasibleClients), 'total_num_samples': self.feasible_samples}

    def getClientReward(self, client_id):
        return self.ucb_sampler.get_client_reward(client_id)

    def get_median_reward(self):
        if self.mode == 'oort':
            return self.ucb_sampler.get_median_reward()
        return 0.
