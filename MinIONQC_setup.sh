#!/bin/bash
# MinIONQC.py Installation Bash Script

# Function to check if a command was successful
check_success() {
    if [ $? -ne 0 ]; then
        echo -e "\e[31mERROR: $1 failed.\e[0m"
        exit 1
    fi
}

# Determine OS Type
OS_TYPE=$(uname -s 2>/dev/null || echo "This pipeline was not built for Windows, please use WSL")

# Checking and Updating Conda
echo -e "\e[36mChecking for Conda...\e[0m"
if ! command -v conda &> /dev/null; then
    echo -e "\e[31mConda is not installed. Please install Conda before running this script.\e[0m"
    exit 1
else
    echo -e "\e[36mUpdating Conda...\e[0m"
    conda update -n base -c defaults conda -y
    check_success "Conda update"
fi

# Checking for Mamba
echo -e "\e[36mChecking for Mamba...\e[0m"
if ! command -v mamba &> /dev/null; then
    echo -e "\e[36mMamba is not installed. Attempting to install Mamba...\e[0m"
    if [ "$OS_TYPE" != "Windows" ]; then
        wget --no-check-certificate https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
        bash Mambaforge-Linux-x86_64.sh
        rm Mambaforge-Linux-x86_64.sh
    else
        # If Mamba not installed
        echo "Please manually install Mambaforge for Windows."
        echo -e "\e[33mFinal Instructions - Close this terminal; open a new one and run the following commands:\e[0m"
        echo -e "\e[36m'mamba install -y -c bioconda -c conda-forge -c agbiome -c prkrekel numpy==1.26.3 pandas==2.2.0 matplotlib==3.8.2 seaborn==0.13.1 pyyaml==6.0.1 statsmodels==0.14.1\e[0m"
        echo -e "\e[32mOnce the above are done Set-up is Complete!\e[0m"
    fi
else
    # Run Mamba installations
    mamba install -y -c bioconda -c conda-forge -c agbiome -c prkrekel numpy==1.26.3 pandas==2.2.0 matplotlib==3.8.2 seaborn==0.13.1 pyyaml==6.0.1 statsmodels==0.14.1
fi

# Give Final Instructions
echo -e "\e[32mMinIONQC.py  set-up done!\e[0m"

echo -e "\e[36mRun 'python MinIONQC.py -i /path/to/summary(s)' to execute the program!\e[0m"

# About Section
echo "@author: ian.michael.bollinger@gmail.com/researchconsultants@critical.consulting"
echo "This is a modified protocol by Roblanf "
echo "GitHub Repository for Original R-Script: https://github.com/roblanf/minion_qc/blob/master/MinIONQC.R"
