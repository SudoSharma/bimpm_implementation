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

# Handle spacy installation
sed -i "/en-core/ d" requirements.txt  # Remove model download
echo "https://github.com/explosion/spacy-models/releases/download/el_core_news_sm-2.1.0a0/el_core_news_sm-2.1.0a0.tar.gz" >> requirements.txt

# Install environment requirements
echo "Installing environment requirements..."
if [ "$COMPUTE" == GPU ] ; then
    pip install -q -r requirements_gpu.txt
else
    pip install -q -r requirements.txt
fi

# link spacy
python -m spacy link en_core_web_sm en --force

echo "Successfully installed environment!"
echo "Activate your environment with 'source activate bimpm'."
