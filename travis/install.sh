#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.

# Deactivate the travis-provided virtual environment and setup a
# conda-based environment instead
deactivate

# Add the miniconda bin directory to $PATH
export PATH=/home/travis/miniconda3/bin:$PATH
echo $PATH

# Use the miniconda installer for setup of conda itself
pushd .
cd
mkdir -p download
cd download
if [[ ! -f /home/travis/miniconda3/bin/activate ]]
then
    if [[ ! -f miniconda.sh ]]
    then
        wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
             -O miniconda.sh
    fi
    chmod +x miniconda.sh && ./miniconda.sh -b -f
    conda config --set always_yes yes
    conda update conda
    echo "Creating environment to run tests in."
    conda create -q -n testenv python=3.6
fi
cd ..
popd

# Activate the python environment we created.
source activate testenv

# Install requirements in our conda environment
echo "Installing environment requirements..."
pip install --upgrade pip
conda install cython -q
conda install plac -q
conda install pytorch-cpu -c pytorch -q
pip install tensorboardX
conda install dill -q
pip install torchtext
conda install spacy -q
pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz"
python -m spacy link en_core_web_sm en --force
