#!/usr/bin/env python
#!/bin/bash

# # Clone the FLOAT repository
# git clone https://github.com/AFKD98/FLOAT.git

# # Change directory to the cloned repository
# cd FLOAT/


# Please replace ~/.bashrc with ~/.bash_profile for MacOS

FLOAT_HOME=$(pwd)

chmod +x $FLOAT_HOME/float.sh

if [[ $(uname -s) == 'Darwin' ]]; then
  echo MacOS
  echo export FLOAT_HOME=$(pwd) >> ~/.bash_profile
  echo alias fedscale=\'bash ${FLOAT_HOME}/float.sh\' >> ~/.bash_profile
  echo alias float=\'bash ${FLOAT_HOME}/float.sh\' >> ~/.bash_profile
  
else
  echo export FLOAT_HOME=$(pwd) >> ~/.bashrc
  echo alias fedscale=\'bash ${FLOAT_HOME}/fedscale.sh\' >> ~/.bashrc
  echo alias float=\'bash ${FLOAT_HOME}/float.sh\' >> ~/.bashrc
fi


isPackageNotInstalled() {
  $1 --version &> /dev/null
  if [ $? -eq 0 ]; then
    echo "$1: Already installed"
  elif [[ $(uname -p) == 'arm' ]]; then
    install_dir=$HOME/miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
    bash  Miniconda3-latest-MacOSX-arm64.sh -b -p  $install_dir
    export PATH=$install_dir/bin:$PATH
  else
    install_dir=$HOME/anaconda3
    wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
    bash Anaconda3-2020.11-Linux-x86_64.sh -b -p  $install_dir
    export PATH=$install_dir/bin:$PATH

  fi
}

# un-comment to install conda
isPackageNotInstalled conda


# create conda env

if [[ $(uname -p) == 'arm' ]]; then
  source ~/miniconda/bin/activate
  . ~/.bash_profile
  conda env create -f environment-arm.yml
  conda install -c apple tensorflow-deps
  conda activate fedscale
  python -m pip install tensorflow-macos==2.9
  python -m pip install tensorflow-metal==0.5.0
  
else
  conda init bash
  . ~/.bashrc
  conda env create -f environment.yml
  conda activate fedscale

fi

#Add authorized key of localhost 
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

if [ "$1" == "--cuda" ]; then
  wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
  sudo apt-get purge nvidia-* -y
  sudo sh -c "echo 'blacklist nouveau\noptions nouveau modeset=0' > /etc/modprobe.d/blacklist-nouveau.conf"
  sudo update-initramfs -u
  sudo sh cuda_10.2.89_440.33.01_linux.run --override --driver --toolkit --samples --silent
  export PATH=$PATH:/usr/local/cuda-10.2/
  conda install cudatoolkit=10.2 -y
fi




# Download the FEMNIST dataset
./benchmark/dataset/download.sh download femnist
