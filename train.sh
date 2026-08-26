#!/bin/bash

# TRAINING PARAMETERS
# dataset='cat'
# dataset='mnist'
dataset='cifar10'

lr=0.001
epochs=30

# TRAINING
# I'm printing the python command just to allow the user to see what is being executed, but the script will launch it anyway.
echo "python3 training.py -dataset $dataset -lr $lr -epochs $epochs"
python3 training.py -dataset $dataset -lr $lr -epochs $epochs
