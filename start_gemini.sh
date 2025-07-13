#!/bin/bash

# Gemini.
env/bin/python3 -m embedding_speed_comparisons \
    -q "p365" \
    --model models/embedding-001 \
    --data_path data/shooting_large.json \
    --vector_size 768 \
    --collection "shooting_large" \
    --gemini \
    --log logs/gemini_benchmark.log \
    -n 10 \
    -r 3
