# BiMPM Implementation in PyTorch
This is a PyTorch implementation of the Bilateral Multi-Perspective Matching for Natural Language Sentences (BiMPM) paper by <em>Wang et al.</em>, which can be found [here](https://arxiv.org/pdf/1702.03814v3.pdf).

# Performance 
## Original Baseline
(Quora) Sentence Similarity: 88.2
<br>(SNLI) Natural Language Inference: 86.9

## Vanilla Reimplementation 
(Quora) Sentence Similarity: 85.9
<br>(SNLI) Natural Language Inference: 85.1

# Requirements
## Python Modules
## System Requirements

# Instructions
## Data
## Folder Structure
## Training
## Evaluation


## Pending Experiments 
- SGDR - stochastic gradient descent with warm restarts
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

# Pending Tasks 
- finish docstrings
- create a requirements file
- add to website, and create UI for Sentence 1, Sentence 2, Inference or Similarity, and provide results - create a command-line script, and a UI
- test "What can I pick up my meds?" and "Where is the nearest pharmacy?"
- write tests
- figure out if CI is important
