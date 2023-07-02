import argparse

from fedscale.cloud import commons


def boolean_string(s):
    if s not in {'False', 'True', '0', '1'}:
        raise ValueError('Not a valid boolean string')
    return (s == 'True' or s != '0')

parser = argparse.ArgumentParser()
parser.add_argument('--job_name', type=str, default='demo_job')
parser.add_argument('--log_path', type=str, default='./',
                    help="default path is ../log")
parser.add_argument('--wandb_token', type=str, default="",
                    help="API key for wandb as login credentials")


#Faraz - new params
parser.add_argument('--exp_type', type=int, default=0) # type of experiment: 0=deadline+target ratio, 1=deadline + no target ratio, 2=overcommit and wait for clients

parser.add_argument('--stale_update', type=int, default=0) # number of rounds to wait for stale updates, 0 inactive, -1
parser.add_argument('--stale_factor', type=float, default=1.0) # multiplication factor for the weight/importance of new updates
parser.add_argument('--straggler_penalty', type=float, default=1.25) # the penalty applied on straggler for utility/score calculation
parser.add_argument('--deadline_percent', type=float, default=0.0) # percent of workers to accept and set as a deadline, either use it or actual deadline
parser.add_argument('--deadline', type=float, default=0.0) # deadline in time for each round, -1 uses last worker time for mov_average, -2 uses target_ratio worker for mov_average
parser.add_argument('--deadline_alpha', type=float, default=0.25) # alpha multiplier on historic values for calculating the moving average of the deadline
parser.add_argument('--deadline_control', type=float, default=0) # use a control logic to adapt the deadline
parser.add_argument('--initial_deadline', type=float, default=1000) # initial deadline if deadline is < 0, i.e., for moving average
parser.add_argument('--target_ratio', type=float, default=0.0) # target percentage of the selected clients to successed for a round to finish
parser.add_argument('--last_worker', type=boolean_string, default=False)  # enables the calculation of round duration based on last worker completion instead waiting for deadline
parser.add_argument('--no_filter_dropouts', type=int, default=0)  # number of times to inflate round duration to wait for dropout clients
parser.add_argument('--avail_priority', type=int, default=0) # priority based on avaliability scheme (1,2,3) - check client_manager
parser.add_argument('--random_behv', type=int, default=0) # assignment of behaviour profile to the clients, 0=sequential, 1=random, 2=assign among least availiable 10%, -1 disable online user behaviour
parser.add_argument('--total_clients', type=int, default=0) # total number of clients to use in simulation, 0 uses the number clients in data map file
parser.add_argument('--used_samples', type=int, default=0) # number of samples per client when partitioning is not based on map file
parser.add_argument('--partitioning', type=int, default=0) # data partitioning scheme among the clients, 1 = uniform, 2 = zipf, 3 = balanced
parser.add_argument('--filter_class', type=int, default=0) # number of classes/labels to randomly filter out (remove) per client
parser.add_argument('--filter_class_ratio', type=float, default=0) # ratio of classes/labels to randomly keep per client
parser.add_argument('--zipf_param', type=float, default=1.95) # the parameter for zipf distribution used by partitioning scheme
parser.add_argument('--dirichlet_alpha', type=float, default=0.1) # the parameter for dirichlet alpha distribution used by partitioning scheme
parser.add_argument('--process_files_ratio', type=float, default=1.0) # percentage of files to process for nlp tasks
parser.add_argument('--train_ratio', type=float, default=1.0) # percentage of train samples per client (partition)
parser.add_argument('--use_wandb', type=boolean_string, default=False) # enables use of wandb logging
parser.add_argument('--resume_wandb', type=boolean_string, default=False) # enables resume of wandb loggging from last point - not sure if it works
parser.add_argument('--wandb_key', type=str, default='') # override if the WANDB_API_KEY is used instead of login
parser.add_argument('--wandb_tags', type=str, default='') # override if the tag is applied for the run
parser.add_argument('--wandb_entity', type=str, default='refl') # The entity or team used for WANDB logging, should be set correctly
parser.add_argument('--mode', type=str, default='async') # mode of aggregation (async, sync, or Fedbuff)
parser.add_argument('--enforce_sgd', type=boolean_string, default=False) # enforce SGD optimizer instead of a custom optimizer per task
parser.add_argument('--dropout_ratio', type=float, default=0.0) # percentage of clients to select at random as dropout clients
parser.add_argument('--send_delta', type=boolean_string, default=False) # send delta of the updated model rather than the updated model
parser.add_argument('--model_boost', type=boolean_string, default=False) # run the stale clients if the model boost was low in the last round
parser.add_argument('--stale_all', type=boolean_string, default=False) # run all clients except one as stale
#parser.add_argument('--use_cached_stale', type=boolean_string, default=False) # skip rounds when we have enough updates from stale clients
parser.add_argument('--adapt_selection', type=int, default=0) # adjust the number of new clients to select, if we have updates from stale clients, 1 = soft adaption, 2 = hard adaptaiton
parser.add_argument('--adapt_selection_cap', type=float, default=0.5) # adjust the number of new clients to select, if we have updates from stale clients, 1 = soft adaption, 2 = hard adaptaiton
parser.add_argument('--initial_total_worker', type=int, default=100) # initial number of total workers to invoke per round
parser.add_argument('--total_worker', type=int, default=5) # number of total workers to invoke per round
parser.add_argument('--buffer_size', type=int, default=10) # Buffer size of Fedbuff
parser.add_argument('--avail_probability', type=float, default=0.0) # probability for the oracle to get the availability right (Accuraccy level)
parser.add_argument('--stale_skip_round', type=boolean_string, default=False) # use the stale updates in rounds with no new updates (skip round)
parser.add_argument('--bandwidth_profiles_dir', type=str, default='') # use the stale updates in rounds with no new updates (skip round)
parser.add_argument('--stale_beta', type=float, default=0.35) # the beta value used for the weigthed average of boosting and damping of the stale updates
parser.add_argument('--scale_coff', type=float, default=10) #The scaling coefficient to boost the good stale updates
parser.add_argument('--adapt_scale_coff', type=boolean_string, default=False) # wether we adapt the scaling coefficient, disabled by default

parser.add_argument('--prohibit_reselect', type=int, default=5) # prohibit the selection of the same client for certain number of round

parser.add_argument('--scale_sys_percent', type=float, default=0.0) # Percentage of clients to scale their system configurations of the clients to make them more or less powerful
parser.add_argument('--scale_sys', type=float, default=0.0) # By how much to scale up or down the client system (e.g., if set to 2 the devices compute capabilities are doubled)

# The basic configuration of the cluster
parser.add_argument('--ps_ip', type=str, default='127.0.0.1')
parser.add_argument('--ps_port', type=str, default='29500')
parser.add_argument('--this_rank', type=int, default=1)
parser.add_argument('--connection_timeout', type=int, default=60)
parser.add_argument('--experiment_mode', type=str,
                    default=commons.SIMULATION_MODE)
parser.add_argument('--engine', type=str, default=commons.PYTORCH,
                    help="Tensorflow or Pytorch for cloud aggregation")
parser.add_argument('--num_executors', type=int, default=1)
parser.add_argument('--executor_configs', type=str,
                    default="127.0.0.1:[1]")  # seperated by ;
# Note: In async mode, the num_participants param is treated as the async buffer size. In sync, this is the number
# of clients that are selected each round.
parser.add_argument('--num_participants', type=int, default=4)
parser.add_argument('--data_map_file', type=str, default=None)
parser.add_argument('--use_cuda', type=str, default='True')
parser.add_argument('--cuda_device', type=str, default=None)
parser.add_argument('--time_stamp', type=str, default='logs')
parser.add_argument('--task', type=str, default='cv')
parser.add_argument('--device_avail_file', type=str, default=None)
parser.add_argument('--clock_factor', type=float, default=1.0,
                    help="Refactor the clock time given the profile")

# The configuration of model and dataset
parser.add_argument('--model_zoo', type=str, default='torchcv',
                    help="model zoo to load the models from", choices=["torchcv", "fedscale-torch-zoo",
                                                                       "fedscale-tensorflow-zoo"])
parser.add_argument('--data_dir', type=str, default='~/cifar10/')
parser.add_argument('--device_conf_file', type=str, default='/tmp/client.cfg')
parser.add_argument('--model', type=str, default='shufflenet_v2_x2_0')
parser.add_argument('--data_set', type=str, default='cifar10')
parser.add_argument('--sample_mode', type=str, default='random')
parser.add_argument('--filter_less', type=int, default=32)
parser.add_argument('--filter_more', type=int, default=1e15)
parser.add_argument('--train_uniform', type=bool, default=False)
parser.add_argument('--conf_path', type=str, default='~/dataset/')
parser.add_argument('--overcommitment', type=float, default=1.3)
parser.add_argument('--model_size', type=float, default=65536)
parser.add_argument('--round_threshold', type=float, default=30)
parser.add_argument('--round_penalty', type=float, default=2.0)
parser.add_argument('--clip_bound', type=float, default=0.9)
parser.add_argument('--blacklist_rounds', type=int, default=-1)
parser.add_argument('--blacklist_max_len', type=float, default=0.3)
parser.add_argument('--embedding_file', type=str,
                    default='glove.840B.300d.txt')
parser.add_argument('--input_shape', type=int, nargs='+', default=[1, 3, 28, 28])
parser.add_argument('--save_checkpoint', type=bool, default=True)


# The configuration of different hyper-parameters for training
parser.add_argument('--rounds', type=int, default=50)
parser.add_argument('--local_steps', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=30)
parser.add_argument('--test_bsz', type=int, default=128)
parser.add_argument('--backend', type=str, default="gloo")
parser.add_argument('--learning_rate', type=float, default=5e-2)
parser.add_argument('--min_learning_rate', type=float, default=5e-5)
parser.add_argument('--input_dim', type=int, default=0)
parser.add_argument('--output_dim', type=int, default=0)
parser.add_argument('--dump_epoch', type=int, default=1e10)
parser.add_argument('--decay_factor', type=float, default=0.98)
parser.add_argument('--decay_round', type=float, default=10)
parser.add_argument('--num_loaders', type=int, default=2)
parser.add_argument('--eval_interval', type=int, default=5)
parser.add_argument('--sample_seed', type=int, default=233)  # 123 #233
parser.add_argument('--test_ratio', type=float, default=1.0)
parser.add_argument('--val_ratio', type=float, default=0.001)
parser.add_argument('--loss_decay', type=float, default=0.2)
parser.add_argument('--exploration_min', type=float, default=0.3)
parser.add_argument('--cut_off_util', type=float,
                    default=0.05)  # 95 percentile

parser.add_argument('--gradient_policy', type=str, default=None)

# for yogi
parser.add_argument('--yogi_eta', type=float, default=3e-3)
parser.add_argument('--yogi_tau', type=float, default=1e-8)
parser.add_argument('--yogi_beta', type=float, default=0.9)
parser.add_argument('--yogi_beta2', type=float, default=0.99)


# for prox
parser.add_argument('--proxy_mu', type=float, default=0.1)

# for detection
parser.add_argument('--cfg_file', type=str,
                    default='./utils/rcnn/cfgs/res101.yml')
parser.add_argument('--test_output_dir', type=str, default='./logs/server')
parser.add_argument('--train_size_file', type=str, default='')
parser.add_argument('--test_size_file', type=str, default='')
parser.add_argument('--data_cache', type=str, default='')
parser.add_argument('--backbone', type=str, default='./resnet50.pth')


# for malicious
parser.add_argument('--malicious_factor', type=int, default=1e15)

# for asynchronous FL
parser.add_argument('--max_concurrency', type=int, default=10)
parser.add_argument('--max_staleness', type=int, default=5)

# for differential privacy
parser.add_argument('--noise_factor', type=float, default=0.1)
parser.add_argument('--clip_threshold', type=float, default=3.0)
parser.add_argument('--target_delta', type=float, default=0.0001)

# for Oort
parser.add_argument('--pacer_delta', type=float, default=5)
parser.add_argument('--pacer_step', type=int, default=20)
parser.add_argument('--exploration_alpha', type=float, default=0.3)
parser.add_argument('--exploration_factor', type=float, default=0.9)
parser.add_argument('--exploration_decay', type=float, default=0.98)
parser.add_argument('--sample_window', type=float, default=5.0)

# for albert
parser.add_argument(
    "--line_by_line",
    action="store_true",
    help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
)
parser.add_argument('--clf_block_size', type=int, default=32)


parser.add_argument(
    "--mlm", type=bool, default=False, help="Train with masked-language modeling loss instead of language modeling."
)
parser.add_argument(
    "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
)
parser.add_argument(
    "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
)
parser.add_argument(
    "--block_size",
    default=64,
    type=int,
    help="Optional input sequence length after tokenization."
    "The training dataset will be truncated in block of this size for training."
    "Default to the model max input length for single sentence inputs (take into account special tokens).",
)


parser.add_argument("--weight_decay", default=0, type=float,
                    help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8,
                    type=float, help="Epsilon for Adam optimizer.")

# for tag prediction
parser.add_argument("--vocab_token_size", type=int,
                    default=10000, help="For vocab token size")
parser.add_argument("--vocab_tag_size", type=int,
                    default=500, help="For vocab tag size")

# for rl example
parser.add_argument("--epsilon", type=float, default=0.9, help="greedy policy")
parser.add_argument("--gamma", type=float, default=0.9, help="reward discount")
parser.add_argument("--memory_capacity", type=int,
                    default=2000, help="memory capacity")
parser.add_argument("--target_replace_iter", type=int,
                    default=15, help="update frequency")
parser.add_argument("--n_actions", type=int, default=2, help="action number")
parser.add_argument("--n_states", type=int, default=4, help="state number")


parser.add_argument("--num_classes", type=int, default=35,
                    help="For number of classes of the dataset")


# for voice
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--test-manifest', metavar='DIR',
                    help='path to test manifest csv', default='data/test_manifest.csv')
parser.add_argument('--sample-rate', default=16000,
                    type=int, help='Sample rate')
parser.add_argument('--labels-path', default='labels.json',
                    help='Contains all characters for transcription')
parser.add_argument('--window-size', default=.02, type=float,
                    help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float,
                    help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming',
                    help='Window type for spectrogram generation')
parser.add_argument('--hidden-size', default=256,
                    type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=7,
                    type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='lstm',
                    help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Finetune the model from checkpoint "continue_from"')
parser.add_argument('--speed-volume-perturb', dest='speed_volume_perturb', action='store_true',
                    help='Use random tempo and gain perturbations.')
parser.add_argument('--spec-augment', dest='spec_augment', action='store_true',
                    help='Use simple spectral augmentation on mel spectograms.')
parser.add_argument('--noise-dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=0.4,
                    help='Probability of noise being added per sample')
parser.add_argument('--noise-min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True,
                    help='Turn off bi-directional RNNs, introduces lookahead convolution')

args, unknown = parser.parse_known_args()
args.use_cuda = eval(args.use_cuda)


datasetCategories = {'Mnist': 10, 'cifar10': 10, "imagenet": 1000, 'emnist': 47,
                     'openImg': 596, 'google_speech': 35, 'femnist': 62, 'yelp': 5
                     }

# Profiled relative speech w.r.t. Mobilenet
model_factor = {'shufflenet': 0.0644/0.0554,
                'albert': 0.335/0.0554,
                'resnet': 0.135/0.0554,
                }

args.num_class = datasetCategories.get(args.data_set, args.num_classes)
for model_name in model_factor:
    if model_name in args.model:
        args.clock_factor = args.clock_factor * model_factor[model_name]
        break
