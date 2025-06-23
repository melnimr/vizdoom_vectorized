#!/bin/bash

for n in  10 30 40 50 75 100 250 500 1000 ; do
    python3.10 gym_vectorized_no_learning.py --n_envs $n
done
