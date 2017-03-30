# Recurrent Neural Network for TV Script generation (work in progress)

I built a Recurrent Neural Network (i.e. RNN) that can be used to generate new TV scripts for the Simpsons show with a loss rate of .004 My dataset consists of a subset of the Simpsons dataset of scripts from 27 seasons.

The RNN is built on TensorFlow, written in Python 3 and is presented via Jupyter Notebook. The RNN was trained on a cloud-based GPU using FloydHub. 


The following are some of the steps I took to build this RNN:

Preprocessing
- Created a Lookup Table with two dictionaries (Word to ID and ID to Word) used for word embeddings
- Split scripts into word arrays and implemented a function for tokenizing punctuation. The punctuation becomes like another word in the word array. This makes it easier for the RNN to predict the next word. 

Build the Neural Network: Implemented the following functions as core components for building the RNN
- get_inputs: creates TF Placeholders for inputs, targets, and learning rate in the Neural Network
- get_init_cell: build RNN cell and initialize; Stacked multiple LSTM layers with tf.contrib.rnn.MultiRNNCell
- get_embed: Applied word embedding to input_data, Return the embedded sequence.
- build_rnn: Build the RNN using tf.nn.dynamic_rnn()
- build_nn: Build the NN by calling functions get_embed, build_rnn. Apply FC layer with linear activation. Return logits, final_state.
- get_batches: Create batches of input and targets as a Numpy array with shape (num_batches, 2, batch_size, seq_length)

Training the network
- Hyperparameters: epochs, batch size, rnn size, sequence length, learning rate
- Training: Trained the neural network on the preprocessed data. Achieved loss rate of 0.004 after 120 epochs. 

Generate Script
- get_tensors: Get tensors from loaded_graph
- pick_word: function to select the next word using probabilities.
- Generate TV Script!
