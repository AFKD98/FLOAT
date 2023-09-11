# -*- coding: utf-8 -*-
import csv
import logging
import random
import time
from collections import Counter, defaultdict
from random import Random
# Faraz - add new modules
from collections import OrderedDict

import numpy as np
from torch.utils.data import DataLoader
import os
import pickle
#from argParser import args


class Partition(object):
    """ Dataset partitioning helper """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """Partition data by trace or random"""

    def __init__(self, data, args, numOfClass=0, seed=10, isTest=False, isVal=False):
        try:
            self.partitions = []
            self.rng = Random()
            self.rng.seed(seed)

            self.data = data
            self.labels = self.data.targets
            self.args = args
            self.isTest = isTest
            self.isVal = isVal
            np.random.seed(seed)

            self.data_len = len(self.data)
            self.numOfLabels = numOfClass
            self.client_label_cnt = defaultdict(set)
            
            #Faraz - set the number of samples per worker
            self.usedSamples = 0

            #Faraz - introduce targets dict
            self.targets = OrderedDict()
            self.indexToLabel = {}

            # categarize the samples
            # last_label = None
            # count = 0
            for index, label in enumerate(self.labels):
                if label not in self.targets:
                    self.targets[label] = []

                self.targets[label].append(index)
                self.indexToLabel[index] = label
        except Exception as e:
            print("Error in DataPartitioner: ", e)

    def getNumOfLabels(self):
        return self.numOfLabels

    def getDataLen(self):
        return self.data_len
    
    def get_number_samples_per_client(self, client_id):
        return len(self.partitions[client_id])

    def getClientLen(self):
        return len(self.partitions)

    def getClientLabel(self):
        return [len(self.client_label_cnt[i]) for i in range(self.getClientLen())]

    def trace_partition(self, data_map_file, ratio=1.0):
        """Read data mapping from data_map_file. Format: <client_id, sample_name, sample_category, category_id>"""
        logging.info(f"Partitioning data by profile {data_map_file}...")

        client_id_maps = {}
        unique_client_ids = {}
        # load meta data from the data_map_file
        with open(data_map_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            read_first = True
            sample_id = 0

            for row in csv_reader:
                if read_first:
                    logging.info(f'Trace names are {", ".join(row)}')
                    read_first = False
                else:
                    client_id = row[0]

                    if client_id not in unique_client_ids:
                        unique_client_ids[client_id] = len(unique_client_ids)

                    client_id_maps[sample_id] = unique_client_ids[client_id]
                    self.client_label_cnt[unique_client_ids[client_id]].add(
                        row[-1])
                    sample_id += 1

        # Partition data given mapping
        self.partitions = [[] for _ in range(len(unique_client_ids))]

        for idx in range(sample_id):
            self.partitions[client_id_maps[idx]].append(idx)
            
        for i in range(len(unique_client_ids)):
            self.rng.shuffle(self.partitions[i])
            takelen = max(0, int(len(self.partitions[i]) * ratio))
            self.partitions[i] = self.partitions[i][:takelen]
            
    # #Faraz - add data mapping handlers (uniform, zipf, balanced) and class exclusion
    # def partition_data_helper(self, num_clients, data_map_file=None):

    #     # read mapping file to partition trace
    #     if data_map_file is not None:
    #         self.trace_partition(data_map_file)
    #     else:
    #         self.uniform_partition(num_clients=num_clients)
    #Faraz - add data mapping handlers (uniform, zipf, balanced) and class exclusion
    def partition_data_helper(self, num_clients, data_map_file=None):
        try:
            tasktype = 'train' if not self.isTest else 'test'
            data_map_file = None
            if data_map_file is not None:
                data_map_file = os.path.join(data_map_file, tasktype + '.csv')
                #Faraz - handle the case for reddit dataset where on IBEX mappings are stored on the metadata folder
                if self.args.data_set == 'reddit' or self.args.data_set == 'stackoverflow':
                    data_map_file = os.path.join(self.args.log_path, 'metadata', self.args.data_set, tasktype)
                    data_map_file = os.path.join(data_map_file,  'result_' + str(self.args.process_files_ratio) + '.csv')

            # Faraz - apply ratio on the data - manipulate the data per uses
            ratio = 1.0
            if not self.isTest and self.args.train_ratio < 1.0:
                ratio = self.args.train_ratio
            elif self.isTest and self.args.test_ratio < 1.0:
                ratio = self.args.test_ratio

            # Faraz - introduce the mapping based on other methods rather than read mapping file to partition trace
            if self.isTest:
                if self.args.partitioning < 0 or data_map_file is None or num_clients < self.args.total_worker:
                    self.uniform_partition(num_clients=num_clients, ratio=ratio)
                else:
                    self.trace_partition(data_map_file, ratio=ratio)
            elif self.args.partitioning <= 0:
                if self.args.partitioning < 0 or data_map_file is None:
                    self.uniform_partition(num_clients=num_clients, ratio=ratio)
                else:
                    self.trace_partition(data_map_file, ratio=ratio)
            else:
                logging.info(f"Partitioning data by {self.args.partitioning} custom partitioning...")
                self.custom_partition(num_clients=num_clients, ratio=ratio)
        except Exception as e:
            print("Error in partitioning data: ", e)
            logging.error(f"Error in partitioning data {e}")

    def uniform_partition(self, num_clients, ratio=1.0):
        # random partition
        numOfLabels = self.getNumOfLabels()
        # data_len = self.getDataLen()
        #Faraz - update the data length to account for the ratio
        data_len = min(self.getDataLen(), int(self.getDataLen() * ratio))
        logging.info(f"Uniform partitioning data, ratio: {ratio} applied for {data_len} samples of {numOfLabels} labels on {num_clients} clients ...")
        logging.info(f"Randomly partitioning data, {data_len} samples...")

        indexes = list(range(data_len))
        self.rng.shuffle(indexes)

        for _ in range(num_clients):
            part_len = int(1./num_clients * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]
        #store the partitions as pickle file
        filename = 'part' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
                    + str(numOfLabels) + '_samples' + str(self.usedSamples)
        folder = os.path.join(self.args.data_dir, 'metadata', self.args.data_set, 'data_mappings')
        logging.info('data path: {}'.format(folder))
        if not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)
        custom_mapping_file = os.path.join(folder, filename)
        if not os.path.exists(custom_mapping_file):
            with open(custom_mapping_file, 'wb') as fout:
                pickle.dump(self.partitions, fout)
            logging.info(f'Storing partitioning to file {filename}')
            

    def custom_partition(self, num_clients, ratio=1.0):
        try:
            # custom partition
            numOfLabels = self.getNumOfLabels()
            # Faraz - update the data length to account for the ratio
            data_len = min(self.getDataLen(), int(self.getDataLen() * ratio))
            sizes = [1.0 / num_clients for _ in range(num_clients)]
            #get # of samples per worker
            #Faraz - set the number of samples per worker
            self.usedSamples = self.args.used_samples if self.args.used_samples >= 0 else (self.args.batch_size + 1)
            # get number of samples per worker
            if self.usedSamples <= 0:
                self.usedSamples = int(data_len / num_clients)
            #Verify if the custom client partitioning exists
            num_class = numOfLabels
            num_remove_classes = 0
            if self.args.filter_class > 0:
                num_remove_classes = self.args.filter_class
            elif self.args.filter_class_ratio > 0:
                num_remove_classes = round(numOfLabels * (1 - self.args.filter_class_ratio))
            num_class -= num_remove_classes
            filename = 'part' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
                    + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)
            if self.args.partitioning == 4 and '_alpha' not in filename:
                filename += '_alpha{}'.format(self.args.dirichlet_alpha)
            folder = os.path.join(self.args.data_dir, 'metadata', self.args.data_set, 'data_mappings')
            logging.info('data path: {}'.format(folder))
            if not os.path.isdir(folder):
                os.makedirs(folder, exist_ok=True)
            custom_mapping_file = os.path.join(folder, filename)
            if self.args.this_rank != 1:
                while (not os.path.exists(custom_mapping_file)):
                    time.sleep(120)
            if os.path.exists(custom_mapping_file):
                with open(custom_mapping_file, 'rb') as fin:
                    logging.info(f'Loading partitioning from file {filename}')
                    self.partitions = pickle.load(fin)
                    for i, part in enumerate(self.partitions):
                        labels = [self.indexToLabel[index] for index in part]
                        logging.info("Found the client mapping file: part {} len: {} labels: {}".format(i, len(part), Counter(labels)))
                        #count_elems = Counter(labels)
                        #logging.info(f'part {i} len: {len(part)} labels: {count_elems.keys()} count: {count_elems.values()}')
                return
            #get targets
            # logging.info("HERE")
            targets = self.getTargets()
            # logging.info("HERE1")
            
            keyDir = {key: int(key) for i, key in enumerate(targets.keys())}
            # logging.info("HERE2")
            
            keyLength = [0] * numOfLabels
            # logging.info("HERE3")
            for key in keyDir.keys():
                keyLength[keyDir[key]] = len(targets[key])

            logging.info(f"Custom partitioning {self.args.partitioning} data, {data_len} samples of {numOfLabels}:{num_class} labels on {num_clients} clients, use {self.usedSamples} sample per client ...")

            ratioOfClassWorker = self.create_mapping(sizes)
            logging.info(f"ratioOfClassWorker: {ratioOfClassWorker}")
            # logging.info("HERE4")

            if ratioOfClassWorker is None:
                return self.uniform_partition(num_clients=num_clients)

            # logging.info("HERE5")

            sumRatiosPerClass = np.sum(ratioOfClassWorker, axis=1)
            for worker in range(len(sizes)):
                ratioOfClassWorker[worker, :] = ratioOfClassWorker[worker, :] / float(sumRatiosPerClass[worker])

            # logging.info("HERE6")

            # classPerWorker -> Rows are workers and cols are classes
            tempClassPerWorker = np.zeros([len(sizes), numOfLabels])

            # logging.info("HERE7")

            # split the classes
            for worker in range(len(sizes)):
                self.partitions.append([])
                # enumerate the ratio of classes it should take
                # logging.info("HERE8")
                for c in list(targets.keys()):
                    takeLength = int(self.usedSamples * ratioOfClassWorker[worker][keyDir[c]])
                    # logging.info("HERE9")
                    
                    takeLength = min(takeLength, keyLength[keyDir[c]])
                    # logging.info("HERE10")

                    indexes = self.rng.sample(targets[c], takeLength)
                    self.partitions[-1] += indexes
                    # logging.info("HERE11")
                    
                    labels = [self.indexToLabel[index] for index in self.partitions[-1]]
                    count_elems = Counter(labels)
                    # logging.info("HERE12")
                    
                    tempClassPerWorker[worker][keyDir[c]] += takeLength


                #logging.info(f'worker: {worker} created partition len: {len(self.partitions[-1])} class/worker: {sum(tempClassPerWorker[worker])} labels:{tempClassPerWorker[worker]} ratios: {ratioOfClassWorker[worker]}')

            del tempClassPerWorker

            #save the partitions as pickle file
            if not os.path.exists(custom_mapping_file):
                # logging.info("HERE19")
                with open(custom_mapping_file, 'wb') as fout:
                    pickle.dump(self.partitions, fout)
                    # logging.info("HERE20")
                logging.info(f'Storing partitioning to file {filename}')
        except Exception as e:
            logging.info(f'Exception in custom partitioning: {e}')
            return self.uniform_partition(num_clients=num_clients)

    def create_mapping(self, sizes):
        try:
            numOfLabels = self.getNumOfLabels()

            ratioOfClassWorker = None
            if self.args.partitioning == 1:
                ratioOfClassWorker = np.random.rand(len(sizes), numOfLabels).astype(np.float32)
            elif self.args.partitioning == 2:
                ratioOfClassWorker = np.random.zipf(self.args.zipf_param, [len(sizes), numOfLabels]).astype(np.float32)
            elif self.args.partitioning == 3:
                ratioOfClassWorker = np.ones((len(sizes), numOfLabels)).astype(np.float32)
            elif self.args.partitioning == 4:
                # Faraz - Generate Dirichlet distribution
                
                ratioOfClassWorker = np.random.dirichlet([self.args.dirichlet_alpha], [len(sizes), numOfLabels]).astype(np.float32)

            # logging.info('alpha: {}'.format(self.args.dirichlet_alpha))
            # logging.info(f"len(sizes): {len(sizes)} numOfLabels: {numOfLabels} ratioOfClassWorker: {ratioOfClassWorker.shape}")
            # logging.info("Dirichlet ratioOfClassWorker: {}".format(ratioOfClassWorker))
            num_remove_class=0
            if self.args.filter_class > 0 or self.args.filter_class_ratio > 0:
                num_remove_class = self.args.filter_class if self.args.filter_class > 0 else round(numOfLabels * (1 - self.args.filter_class_ratio))
                for w in range(len(sizes)):
                    # randomly filter classes by forcing zero samples
                    wrandom = self.rng.sample(range(numOfLabels), num_remove_class)
                    for wr in wrandom:
                        ratioOfClassWorker[w][wr] = 0.0 #0.001

            #logging.info("==== Class per worker partitioning:{} clients:{} labels:{} rem_lable:{} count:{}  ====\n {} \n".format(self.args.partitioning, len(sizes), numOfLabels, num_remove_class, np.count_nonzero(ratioOfClassWorker), repr(ratioOfClassWorker)))
            logging.info("==== Class per worker partitioning:{} clients:{} labels:{} rem_label:{} count:{}  ==== \n".format(self.args.partitioning, len(sizes), numOfLabels, num_remove_class, np.count_nonzero(ratioOfClassWorker)))
            return ratioOfClassWorker
        except:
            logging.info("Error in creating mapping")
            return None

    def getTargets(self):
        tempTarget = self.targets.copy()
        #TODO:why the temp targets are reshuffled each time getTargets is called?
        for key in tempTarget:
             self.rng.shuffle(tempTarget[key])
        return tempTarget

    def log_selection(self,classPerWorker):
        totalLabels = [0 for i in range(len(classPerWorker[0]))]
        logging.info("====Total # of workers is :{}, w/ {} labels, {}".format(len(classPerWorker), len(classPerWorker[0]), len(self.partitions)))
        for index, row in enumerate(classPerWorker):
            rowStr = ''
            numSamples = 0
            for i, label in enumerate(classPerWorker[index]):
                rowStr += '\t'+str(int(label))
                totalLabels[i] += label
                numSamples += label
            logging.info(str(index) + ':\t' + rowStr + '\t' + 'with sum:\t' + str(numSamples) + '\t' + repr(len(self.partitions[index])))
            logging.info("=====================================\n")
        logging.info("Total selected samples is: {}, with {}\n".format(str(sum(totalLabels)), repr(totalLabels)))
        logging.info("=====================================\n")
        
    def use(self, partition, istest, isVal):
        try:
            resultIndex = self.partitions[partition % len(self.partitions)]
            exeuteLength = int(len(resultIndex) * (1 - self.args.val_ratio)) if not istest else int(
                len(resultIndex) * self.args.test_ratio)
            resultIndex = resultIndex[:exeuteLength]
            if isVal:
                exeuteLength = int(len(resultIndex) * self.args.val_ratio)
                resultIndex = resultIndex[-exeuteLength:]
                # logging.info("Faraz (data) - Partition: {} with {} samples executeLength {} and isTest {} and isVal {}".format(partition, len(resultIndex), exeuteLength, istest, isVal))
            self.rng.shuffle(resultIndex)
            # logging.info("Faraz (data) - Partition: {} with {} samples and isTest {}".format(partition, len(resultIndex), istest))

            return Partition(self.data, resultIndex)
        except Exception as e:
            logging.error(f'Exception in use: {e}')
            return None

    def getSize(self):
        # return the size of samples
        return {'size': [len(partition) for partition in self.partitions]}


def select_dataset(rank, partition, batch_size, args, isTest=False, isVal=False, collate_fn=None):
    """Load data given client Id"""
    try:
        
        partition = partition.use(rank - 1, isTest, isVal)
        dropLast = False if isTest else True
        if isTest:
            num_loaders = 0
        else:
            num_loaders = min(int(len(partition)/args.batch_size/2), args.num_loaders)
        if num_loaders == 0:
            time_out = 0
        else:
            time_out = 60

        if collate_fn is not None:
            return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, timeout=time_out, num_workers=num_loaders, drop_last=dropLast, collate_fn=collate_fn)
        return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, timeout=time_out, num_workers=num_loaders, drop_last=dropLast)
    except Exception as e:
        logging.error(f'Exception in select_dataset: {e}')
        return None
