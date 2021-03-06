#!/bin/bash
if [ "$1" == --gpu ] ; then
    echo "Setting up GPU environment..."
    COMPUTE="GPU"
else
    echo "Setting up CPU environment..."
    COMPUTE="CPU"
fi

# Be sure to change the 'miniconda3' to your flavor of Anaconda distribution
export CONDA="miniconda3"
export PATH="$HOME/$CONDA/bin:$PATH"
echo "Setting up 'bimpm' conda environment..."
conda config --set always_yes yes
conda update -q conda
conda create -q -n bimpm python=3.6
source activate bimpm 

# Install environment requirements
pip install --upgrade pip

echo "Installing environment requirements..."
if [ "$COMPUTE" == GPU ] ; then
    conda install pytorch -c pytorch --yes -q
    pip install tensorflow-gpu
else
    conda install pytorch-cpu -c pytorch --yes -q
    pip install tensorflow
fi

conda install cython -q
conda install plac -q
pip install tensorboardX
conda install dill -q
pip install torchtext
conda install spacy -q
pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz"
python -m spacy link en_core_web_sm en --force

echo "Successfully installed environment!"
echo "Activate your environment with 'source activate bimpm'."
