#!/bin/bash


# BERT mini.
env/bin/python3 -m embedding_speed_comparisons \
    -q "p365" \
    --model all-MiniLM-L6-v2 \
    --data_path data/shooting_large.json \
    --vector_size 384 \
    --collection "shooting_large_minilm" \
    --log logs/minilm_benchmark.log \
    -n 10 \
    -r 3