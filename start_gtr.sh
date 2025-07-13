#!/bin/bash




#BERT gtr.
env/bin/python3 -m embedding_speed_comparisons \
    -q "p365" \
    --model sentence-transformers/gtr-t5-xxl \
    --data_path data/shooting_large.json \
    --vector_size 768 \
    --collection "shooting_large_gtr" \
    --log logs/gtr_benchmark.log \
    -n 10 \
    -r 3