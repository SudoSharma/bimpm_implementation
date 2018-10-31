#!/bin/bash
if [ "$1" == --gpu ] ; then
    echo "Setting up GPU conda environment."
    COMPUTE="GPU"
else
    echo "Setting up CPU conda environment"
    COMPUTE="CPU"
fi

# Be sure to change the 'miniconda3' to your flavor of Anaconda distribution
export PATH="$HOME/miniconda3/bin:$PATH"
echo "Setting up bimpm conda environment..."
conda config --set always_yes yes
conda update -q conda
conda create -q -n bimpm python=3.6
conda activate bimpm 

# Handle spacy installation
conda install -q -c conda-forge spacy
python -m spacy download en

# Install environment requirements
echo "Installing environment requirements..."
if [ "$COMPUTE" == GPU ] ; then
    pip install -q -r requirements_gpu.txt
else
    pip install -q -r requirements.txt
fi
