#!/bin/bash


# grab data from cifar-10:

echo "downloading cifar-10 dataset"
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

echo "uncompressing"
tar -xvf cifar-10-python.tar.gz