# BiMPM Implementation in PyTorch
This is a PyTorch implementation of the Bilateral Multi-Perspective Matching for Natural Language Sentences (BiMPM) paper by <em>Wang et al.</em>, which can be found [here](https://arxiv.org/pdf/1702.03814v3.pdf).

# TODOs
- create data loader using pytorch data module
- consider storing tensors in contiguous memory 
- consider adding protection against exploding cosine similarity for very small norm
- create training script
- create test script
- test on cpu - then port code to gpu
- optimize on Quora dataset
- optimize on SNLI dataset
- create benchmark in readme
