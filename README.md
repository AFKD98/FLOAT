<h1> FLOAT: Federated Learning Optimizations with Automated Tuning </h1>

**FLOAT (Federated Learning Optimizations with Automated Tuning) is an advanced framework designed to improve federated learning (FL) by optimizing resource utilization and model performance.**

It addresses the challenges of client heterogeneity, stragglers, and dropouts in FL environments through dynamic resource-aware client optimization strategies. FLOAT leverages a multi-objective Reinforcement Learning with Human Feedback (RLHF) mechanism to automate the selection and configuration of optimization techniques tailored to individual client resource conditions. This approach not only enhances model convergence and performance but also increases accuracy, reduces client dropouts, and improves resource efficiency across communication, computation, and memory usage. Additionally, FLOAT is compatible with existing FL systems and supports both asynchronous and synchronous FL settings, showcasing its versatility and non-intrusiveness. 

**FLOAT is built on FedScale, a scalable and extensible open-source federated learning (FL) engine and benchmark**. 

FedScale ([fedscale.ai](https://fedscale.ai/)) provides high-level APIs to implement FL algorithms, deploy and evaluate them at scale across diverse hardware and software backends. 
FedScale also includes the largest FL benchmark that contains FL tasks ranging from image classification and object detection to language modeling and speech recognition. 
Moreover, it provides datasets to faithfully emulate FL training environments where FL will realistically be deployed.


## Getting Started

### Quick Installation (Linux)

You can simply run `float_install.sh`.

```
source float_install.sh # Add `--cuda` if you want CUDA 
pip install -r requirements.txt && pip install -e .
```

Update `float_install.sh` if you prefer different versions of conda/CUDA. Running **float_install.sh** should install all dependencies in a conda environment and also activate the `fedscale` conda environment on the bash terminal.

### Installation from Source (Linux/MacOS)

If you have [Anaconda](https://www.anaconda.com/products/distribution#download-section) installed and cloned FedScale, here are the instructions.
```
cd FLOAT

# Please replace ~/.bashrc with ~/.bash_profile for MacOS
FLOAT_HOME=$(pwd)
echo export FLOAT_HOME=$(pwd) >> ~/.bashrc
echo alias fedscale=\'bash $FLOAT_HOME/float.sh\' >> ~/.bashrc
echo alias float=\'bash $FLOAT_HOME/float.sh\' >> ~/.bashrc 
conda init bash
. ~/.bashrc

conda env create -f environment.yml
conda activate fedscale
pip install -r requirements.txt && pip install -e .
```

Finally, install NVIDIA [CUDA 10.2](https://developer.nvidia.com/cuda-downloads) or above if you want to use FedScale with GPU support.


### Tutorials

Now that you have FLOAT installed on FedScale, you can start exploring FLOAT following one of these introductory tutorials.

1. [Explore FedScale datasets](./docs/Femnist_stats.md)
2. [Deploy your FL experiment](./docs/tutorial.md)
3. [Implement an FL algorithm](./examples/README.md)
4. [Deploy FL on smartphones](./fedscale/edge/android/README.md)

## FedScale Datasets

***We are adding more datasets! Please contribute!***

FedScale consists of 20+ large-scale, heterogeneous FL datasets and 70+ various [models](./fedscale/utils/models/cv_models/README.md), covering computer vision (CV), natural language processing (NLP), and miscellaneous tasks. 
Each one is associated with its training, validation, and testing datasets. 
We acknowledge the contributors of these raw datasets. Please go to the `./benchmark/dataset` directory and follow the dataset [README](./benchmark/dataset/README.md) for more details.
The datasets can be downloaded using the following command:

```bash
./benchmark/dataset/download.sh download DATASET
```

## FLOAT Runtime
FLOAT Runtime is a scalable and extensible deployment built on FedSCale. 

Please go to `./fedscale/cloud` directory and follow the [README](./fedscale/cloud/README.md) to set up FL training scripts and the [README](./fedscale/edge/android/README.md) for practical on-device deployment.

### Hardware Dependencies

Running experiments do not mandate any special hardware. However, to run the experiments in a reasonable amount of time servers with fast Nvidia GPUs (e.g., A100/V100) or at least 3070 GPUs are recommended. However, due to the scale of the experiments conducted in this study, it may not be feasible to reproduce it due to the large cost incurred. To give an estimate, even with advanced GPUs such as RTX 3070 GPUs, it took a significant amount of time to run them (i.e., 1400 hours of GPU time). This makes it quite hard to reproduce the claims/figures within the time frame set for evaluation.

### Software Dependencies

The FLOAT framework's operation requires Python for core programming, Anaconda for package and environment management, and CUDA for GPU support in accelerated computing tasks. Essential packages and libraries required for FLOAT are included in the **environment.yml** file and **requirements.txt** within the FLOAT repository.

### Benchmarks

FLOAT supports various FL tasks including image classification and speech recognition. Other tasks such as language modeling can also be added as they are supported by FedScale and FLOAT leverages FedScale's extensive dataset and benchmark suite. To provide a realistic simulation environment, FLOAT includes real-world traces for compute, network, and client availability. These traces are critical for accurately simulating FL environments and are located in the following directory of FLOAT's GitHub repository:

## Setup

### Installation

**NOTE:** Although FLOAT automatically finds the correct paths using `os` commands in Python. Nevertheless, please ensure that the paths to the code and datasets are consistent across all nodes so that the simulator can find the right path. FLOAT can run on a single node or on multiple nodes that have at least a 3070 Nvidia GPU and the CPU capacity to run at least 30 threads in parallel.

**Quick start:**
After cloning the repo, go to the main directory FLOAT using the following command:

```bash
cd FLOAT
```

First, edit **float_install.sh** script if necessary. Please, uncomment the parts relating to the installation of the Anaconda Package Manager, CUDA 10.2 if they are not already present on the servers. Note, if you prefer different versions of conda and CUDA, please check the comments in **float_install.sh** for details. After editing, run the following commands to prepare the environment:


To execute an example experiment, use:

```bash
bash float_run_exps.sh -d dataset -a algorithm
```

For example, for running Oort this is the command:

```bash
bash float_run_exps.sh -d femnist -a oort
```

For running Oort with FLOAT the command is as follows:

```bash
bash float_run_exps.sh -d femnist -a oort_float
```

Further commands can be accessed using:

```bash
bash float_run_exps.sh -h
```

## Repo Structure

```
Repo Root
|---- fedscale          # FedScale source code
  |---- cloud           # Core of FedScale service
  |---- utils           # Auxiliaries (e.g, model zoo and FL optimizer)
  |---- edge            # Backends for practical deployments (e.g., mobile)
  |---- dataloaders     # Data loaders of benchmarking dataset

|---- docker            # FedScale docker and container deployment (e.g., Kubernetes)
|---- benchmark         # FedScale datasets and configs
  |---- dataset         # Benchmarking datasets
  |---- configs         # Example configurations

|---- scripts           # Scripts for installing dependencies
|---- examples          # Examples of implementing new FL designs
|---- docs              # FedScale tutorials and APIs
```

## References
Please read and/or cite as appropriate to use FedScale code or data or learn more about FedScale.

```bibtex
@inproceedings{Khan2024FLOAT,
  title={FLOAT: Federated Learning Optimizations with Automated Tuning},
  author={Ahmad Faraz Khan and Azal Ahmad Khan and Ahmed M. Abdelmoniem and Samuel Fountain and Ali R. Butt and Ali Anwar},
  booktitle={Eighteenth European Conference on Computer Systems (EuroSys '24)},
  year={2024},
  address={Athens, Greece},
  pages={1--18},
  doi={10.1145/3552326.3567485}
}
```

and  

```bibtex
@inproceedings{fedscale-icml22,
  title={{FedScale}: Benchmarking Model and System Performance of Federated Learning at Scale},
  author={Fan Lai and Yinwei Dai and Sanjay S. Singapuram and Jiachen Liu and Xiangfeng Zhu and Harsha V. Madhyastha and Mosharaf Chowdhury},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2022}
}
```

## Contributions and Communication
Please submit [issues](https://github.com/AFKD98/FLOAT/issues) or [pull requests](https://github.com/AFKD98/FLOAT/pulls) as you find bugs or improve FLOAT.

For each submission, please add unit tests to the corresponding changes and make sure that all unit tests pass by running `pytest fedscale/tests`.

For any questions or comments, please email us ([afkhan@vt.edu](mailto:afkhan@vt.edu)). 

