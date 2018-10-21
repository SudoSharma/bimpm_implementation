# BiMPM Implementation in PyTorch
This is a PyTorch implementation of the Bilateral Multi-Perspective Matching for Natural Language Sentences (BiMPM) paper by <em>Wang et al.</em>, which can be found [here](https://arxiv.org/pdf/1702.03814v3.pdf).

# TODOs
- finish docstrings
- create a requirements file

# Experiments
Quora Baseline: 88.17 

## Vanilla Version 
Quora Reimplementation: 85.88
SNLI Reimplementation: 

# Ways to improve further on toy Quora dataset
1. SGDR - stochastic gradient descent with warm restarts
- cosine annealing
- add adamw?
- try tanh and then try swish or eswish
- try GRU 
- weight initialization
- bi-lstm char
- more perspectives - try 25
- more hidden neurons in char lstm - 100
- ensembling?
- swap glove vectors with ulmfit word vectors
