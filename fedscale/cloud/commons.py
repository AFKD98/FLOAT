# Define Basic Experiment Setup
from enum import Enum

SIMULATION_MODE = 'simulation'
DEPLOYMENT_MODE = 'deployment'

# Define Basic FL Events
UPDATE_MODEL = 'update_model'
MODEL_TEST = 'model_test'
SHUT_DOWN = 'terminate_executor'
START_ROUND = 'start_round'
CLIENT_CONNECT = 'client_connect'
CLIENT_TRAIN = 'client_train'
CLIENT_VALIDATE = 'client_validate'
CLIENT_VALIDATE_ALL = 'client_validate_all'
DUMMY_EVENT = 'dummy_event'
UPLOAD_MODEL = 'upload_model'

# PLACEHOLD
DUMMY_RESPONSE = 'N'


TENSORFLOW = 'tensorflow'
PYTORCH = 'pytorch'

PRUNING_25 = 'pruning_25'
PRUNING_50 = 'pruning_50'
PRUNING_75 = 'pruning_75'
# PRUNING_95 = 'pruning_95'
QUANTIZATION_8 = 'quantization_8'
QUANTIZATION_16 = 'quantization_16'
PARTIAL_25 = 'partial_25'
PARTIAL_50 = 'partial_25'
PARTIAL_75 = 'partial_75'
# ACTIONS = [QUANTIZATION_8, QUANTIZATION_16, PRUNING_25, PRUNING_50, PRUNING_75, PRUNING_95]
ACTIONS = [QUANTIZATION_8, QUANTIZATION_16, PRUNING_25, PRUNING_50, PRUNING_75, PARTIAL_25, PARTIAL_50, PARTIAL_75]