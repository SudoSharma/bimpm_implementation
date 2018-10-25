# BiMPM Implementation in PyTorch
This is a PyTorch implementation of the Bilateral Multi-Perspective Matching for Natural Language Sentences (BiMPM) paper by <em>Wang et al.</em>, which can be found [here](https://arxiv.org/pdf/1702.03814v3.pdf).

## Model Architecture 

<p align="center">
    <img width="500" src="https://github.com/SudoSharma/bimpm_implementation/blob/master/media/bimpm.png"/>
</p>

# Performance 
## Sentence Similarity
Data: [Quora](https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view) 

| Models        | Accuracy   | 
|--------------|:----------:|
| Original Baseline | 88.2 |
| **Reimplementation** | **85.9** |  

## Natural Language Inference
Data: [SNLI](https://nlp.stanford.edu/projects/snli/)

| Models        |  Accuracy   | 
|--------------|:----------:|
| Original Baseline	| 86.9 |    
| **Reimplementation** | **85.1** |  


# Requirements
## Environment
Please create a new conda environment using the **environment.yml** file.
    
    conda create --name {my_env} -f=environment.yml

## System
- OS: Ubuntu 16.04 LTS (64 bit)
- GPU: 1 x NVIDIA Tesla P100 

# Instructions
If you are SSHing into a cloud instance to train this model, you can modify the **train.sh** shell script with any parameters, and run it using:

    ./train.sh

The script will allow your model to run in the background without interruptions incase your connection ever times out, and the output of the training is stored in `train.out`.

## Training

    $ python train.py -h

    usage: train.py [-h]
                    [batch_size] [char_input_size] [char_hidden_size] [data_type]
                    [dropout] [epoch] [hidden_size] [lr] [num_perspectives]
                    [print_interval] [word_dim]

    Train and store the best BiMPM model in a cycle.

        Parameters
        ----------
        batch_size : int, optional
            Number of examples in one iteration (default is 64).
        char_input_size : int, optional
            Size of character embedding (default is 20).
        char_hidden_size : int, optional
            Size of hidden layer in char lstm (default is 50).
        data_type : {'Quora', 'SNLI'}, optional
            Choose either SNLI or Quora (default is 'quora').
        dropout : int, optional
            Applied to each layer (default is 0.1).
        epoch : int, optional
            Number of passes through full dataset (default is 10).
        hidden_size : int, optional
            Size of hidden layer for all BiLSTM layers (default is 100).
        lr : int, optional
            Learning rate (default is 0.001).
        num_perspectives : int, optional
            Number of perspectives in matching layer (default is 20).
        print_interval : int, optional
            How often to write to tensorboard (default is 500).
        word_dim : int, optional
            Size of word embeddings (default is 300).

        Raises
        ------
        RuntimeError
            If any data source other than SNLI or Quora is requested.

        

    positional arguments:
      batch_size        [64]
      char_input_size   [20]
      char_hidden_size  [50]
      data_type         {[Quora], SNLI}
      dropout           [0.1]
      epoch             [10]
      hidden_size       [100]
      lr                [0.001]
      num_perspectives  [20]
      print_interval    [500]
      word_dim          [300]

    optional arguments:
      -h, --help        show this help message and exit

## Testing 

    $ python test.py -h


    usage: test.py [-h]
                   model_path [batch_size] [char_input_size] [char_hidden_size]
                   [data_type] [dropout] [epoch] [hidden_size] [lr]
                   [num_perspectives] [word_dim]

    Print the best BiMPM model accuracy for the test set in a cycle.

        Parameters
        ----------
        model_path : str
            A path to the location of the BiMPM trained model.
        batch_size : int, optional
            Number of examples in one iteration (default is 64).
        char_input_size : int, optional
            Size of character embedding (default is 20).
        char_hidden_size : int, optional
            Size of hidden layer in char lstm (default is 50).
        data_type : {'Quora', 'SNLI'}, optional
            Choose either SNLI or Quora (default is 'quora').
        dropout : int, optional
            Applied to each layer (default is 0.1).
        epoch : int, optional
            Number of passes through full dataset (default is 10).
        hidden_size : int, optional
            Size of hidden layer for all BiLSTM layers (default is 100).
        lr : int, optional
            Learning rate (default is 0.001).
        num_perspectives : int, optional
            Number of perspectives in matching layer (default is 20).
        word_dim : int, optional
            Size of word embeddings (default is 300).

        Raises
        ------
        RuntimeError
            If any data source other than SNLI or Quora is requested.

        

    positional arguments:
      model_path
      batch_size        [64]
      char_input_size   [20]
      char_hidden_size  [50]
      data_type         {SNLI, [Quora]}
      dropout           [0.1]
      epoch             [10]
      hidden_size       [100]
      lr                [0.001]
      num_perspectives  [20]
      word_dim          [300]

    optional arguments:
      -h, --help        show this help message and exit

# References
1. Wang, Zhiguo, Wael Hamza, and Radu Florian. "Bilateral Multi-Perspective Matching for Natural Language Sentences." Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence, July 14, 2017. Accessed October 10, 2018. doi:10.24963/ijcai.2017/579. 
