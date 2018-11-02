# Ideas
My goal here is to get a good feeling for how the network works and what I can do to make it better. I will experiment in new branches, and then combine everything back into 'develop' and then eventually 'master' if the various network experiments prove fruitful.  

## Research Dataset Sentence Similarity Baseline
First, let's get a baseline for our toy data set. Here's the composition:
Toy Train: 10,000 examples
Toy Dev: 10,000 examples
Toy Test: 10,000 examples

And here are the parameters used:

## Architecture

### Network Weights Initialization

### Character BiLSTM
The authors of the paper used a vanilla LSTM for character-level embeddings. Let's see if a BiLSTM does any better. 

### GRU
Using GRU cell instead of BiLSTM.

### Number of Perspectives

### Increased Hidden Layers in CHar LSTM

### Using ULMFiT Vectors instead of GloVe

## Activation Functions
Experimenting with different activation functions. 

### TanH

### Swish
