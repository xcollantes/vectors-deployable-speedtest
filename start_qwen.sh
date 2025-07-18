#!/bin/bash


# Qwen.
env/bin/python3 -m embedding_speed_comparisons \
    -q "p365" \
    --model Qwen/Qwen3-Embedding-8B \
    --data_path data/shooting_large.json \
    --vector_size 4096 \
    --collection "shooting_large_qwen" \
    --log logs/qwen_benchmark.log \
    -n 10 \
    -r 3


# Gemini.
# env/bin/python3 -m embedding_speed_comparisons \
#     -q "p365" \
#     --model models/embedding-001 \
#     --data_path data/shooting_large.json \
#     --vector_size 768 \
#     --collection "shooting_large" \
#     --gemini \
#     --log logs/gemini_benchmark.log \
#     -n 10 \
#     -r 3
