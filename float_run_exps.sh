#!/bin/bash


# Initialize variables
DATASET=""
ALGORITHM=""

# Function to display the help menu
show_help() {
    echo "Usage: $0 -d dataset -a algorithm"
    echo ""
    echo "Options:"
    echo "  -h, --help           Show this help message"
    echo "  -d dataset           Specify the dataset to use (e.g., femnist)"
    echo "  -a algorithm         Specify the algorithm to use (e.g., fedavg, oort, fedavg_float, fedbuff, oort_float, fedbuff_float)"
    echo ""
    echo "Example:"
    echo "  $0 -d femnist -a fedavg"
}

# the path to the dataset, note $dataset, the dataset name passed as argument to script
export DATA_PATH=/home/ahmad/FLOAT/dataset/data/${dataset}
#the path to the conda envirnoment
export CONDA_ENV=/home/ahmad/anaconda3/envs/fedscale
#the path to the conda source script
export CONDA_PATH=/home/ahmad/anaconda3/

# if [[ $(uname -s) == 'Darwin' ]]; then
#   echo MacOS
#   echo export FLOAT_HOME=$(pwd) >> ~/.bash_profile
#   echo alias fedscale=\'bash ${FLOAT_HOME}/float.sh\' >> ~/.bash_profile
#   echo alias float=\'bash ${FLOAT_HOME}/float.sh\' >> ~/.bash_profile
  
# else
#   echo export FLOAT_HOME=$(pwd) >> ~/.bashrc
#   echo alias fedscale=\'bash ${FLOAT_HOME}/fedscale.sh\' >> ~/.bashrc
#   echo alias float=\'bash ${FLOAT_HOME}/float.sh\' >> ~/.bashrc
# fi


# Parse command line options
while getopts ":hd:a:" opt; do
    case ${opt} in
        h )
            show_help
            # exit 0
            ;;
        d )
            DATASET=$OPTARG
            ;;
        a )
            ALGORITHM=$OPTARG
            ;;
        \? )
            echo "Invalid option: $OPTARG" 1>&2
            show_help
            # exit 1
            ;;
        : )
            echo "Invalid option: $OPTARG requires an argument" 1>&2
            show_help
            # exit 1
            ;;
    esac
done
shift $((OPTIND -1))

# Check if dataset and algorithm have been set
if [ -z "$DATASET" ] || [ -z "$ALGORITHM" ]; then
    echo "Error: Dataset and algorithm must be specified."
    show_help
    # exit 1
fi

# Common command to download the dataset
download_dataset() {
    ./benchmark/dataset/download.sh download $DATASET
}

# # Run experiments based on the algorithm
# run_experiment() {
#     case $ALGORITHM in
#         fedavg)
#             float driver submit benchmark/configs/femnist_fedavg/conf.yml
#             ;;
#         oort)
#             float driver submit benchmark/configs/femnist_oort/conf.yml
#             ;;
#         fedavg_float)
#             float driver submit benchmark/configs/femnist_fedavg_FLOAT_resnet34/conf.yml
#             ;;
#         fedbuff)
#             float driver submit benchmark/configs/fedbuff_femnist/conf.yml
#             ;;
#         oort_float)
#             float driver submit benchmark/configs/femnist_FLOAT_oort_resnet34/conf.yml
#             ;;
#         fedbuff_float)
#             float driver submit benchmark/configs/fedbuff_femnist_FLOAT_resnet34/conf.yml
#             ;;
#         *)
#             echo "Unknown algorithm: $ALGORITHM"
#             # exit 2
#             ;;
#     esac
# }
# Run experiments based on the algorithm
run_experiment() {
    case $ALGORITHM in
        fedavg)
            bash "$FLOAT_HOME/float.sh" driver submit benchmark/configs/femnist_fedavg/conf.yml
            ;;
        oort)
            bash "$FLOAT_HOME/float.sh" driver submit benchmark/configs/femnist_oort/conf.yml
            ;;
        fedavg_float)
            bash "$FLOAT_HOME/float.sh" driver submit benchmark/configs/femnist_fedavg_FLOAT_resnet34/conf.yml
            ;;
        fedbuff)
            bash "$FLOAT_HOME/float.sh" driver submit benchmark/configs/fedbuff_femnist/conf.yml
            ;;
        oort_float)
            bash "$FLOAT_HOME/float.sh" driver submit benchmark/configs/femnist_FLOAT_oort_resnet34/conf.yml
            ;;
        fedbuff_float)
            bash "$FLOAT_HOME/float.sh" driver submit benchmark/configs/fedbuff_femnist_FLOAT_resnet34/conf.yml
            ;;
        *)
            echo "Unknown algorithm: $ALGORITHM"
            # exit 2
            ;;
    esac
}


# Main script execution
download_dataset
run_experiment
