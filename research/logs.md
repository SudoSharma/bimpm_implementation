# Ideas
My goal here is to get a good feeling for how the network works and what I can do to make it better. I will experiment in new branches, and then combine everything back into 'develop' and then eventually 'master' if the various network experiments prove fruitful.  

## Sentence Similarity Baseline
Run: `./xp_0.sh`
Configs

# Experiments 
## Architecture 
ID: `ex_2`

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
