#!/bin/bash
if [ "$1" == --gpu ] ; then
    echo "Setting up GPU environment..."
    COMPUTE="GPU"
else
    echo "Setting up CPU environment..."
    COMPUTE="CPU"
fi

# Be sure to change the 'miniconda3' to your flavor of Anaconda distribution
export PATH="$HOME/miniconda3/bin:$PATH"
echo "Setting up 'bimpm' conda environment..."
conda config --set always_yes yes
conda update -q conda
conda create -q -n bimpm python=3.6
source activate bimpm 

# Remove en_core_sm_md from requirement.
sed -i "/en-core/ d" requirements.txt  # Remove model download

# Install environment requirements
echo "Installing environment requirements..."
if [ "$COMPUTE" == GPU ] ; then
    pip install -q -r requirements_gpu.txt
else
    pip install -q -r requirements.txt
fi

# link spacy
conda install spacy
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz
python -m spacy link en_core_web_sm en --force

echo "Successfully installed environment!"
echo "Activate your environment with 'source activate bimpm'."
