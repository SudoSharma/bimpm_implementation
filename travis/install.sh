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
    conda update --yes conda
    echo "Creating environment to run tests in."
    conda create -n testenv --yes python="$PYTHON_VERSION"
fi
cd ..
popd

# Activate the python environment we created.
source activate testenv

# Install requirements in our conda environment
conda install -q -c conda-forge spacy
python -m spacy download en
pip -q install -r requirements.txt
