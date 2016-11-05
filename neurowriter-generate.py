
# coding: utf-8

# Generates text in the style of a previously learned corpus

from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.models import model_from_json
import numpy as np
import random
import sys
import json

import argparse

# ## Config

# Length of sequences introduced in the LSTM
maxlen = 100

# Parameters declaration
parser = argparse.ArgumentParser(description='Generates text in the style of a previously learned corpus.')
parser.add_argument('modelfile', type=str, help='name of the file with the learned model (without extension)')
parser.add_argument('diversity', type=float, help='diversity value to use in the generation (the higher, the more chaotic)')
parser.add_argument('seed', type=str, help='seed string to use as a starter for the text to write')
parser.add_argument('length', type=int, help='length in chars of the generated text')

# Load parameters
args = parser.parse_args()
modelfile = args.modelfile
diversity = args.diversity
seed = args.seed
length = args.length

# Load model structure
with open(modelfile + "_def.json", "r") as f:
    model = model_from_json(json.load(f))
# Load model weights
print('Loading pre-trained weights...')
model.load_weights(modelfile + ".h5")
# Load character indexes
with open(modelfile + "_idx.json", "r") as f:
    idx = json.load(f)
char_indices = idx["char_indices"]
indices_char = {int(key) : idx["indices_char"][key] 
    for key in idx["indices_char"]}


# Helper function to sample an index from a probability array
#TODO: make this common function with training
def sample(a, temperature=1.0):
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# Helper function to generate text
#TODO: make this common function with training
def gentxt(seed, char_indices, indices_char, diversity, length):
    print(seed, end="")
    # Pad seed with blanks if needed
    if len(seed) < maxlen:
        seed = (" " * (maxlen - len(seed)) ) + seed

    generated = seed

    for i in range(length):
        x = np.zeros((1, maxlen, len(char_indices)))
        for t, char in enumerate(generated[-maxlen:]):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char

        print(next_char, end="", flush=True)
    print("")

# Generate text with the model
gentxt(seed, char_indices, indices_char, diversity, length)

