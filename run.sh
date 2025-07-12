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

# BERT gtr.
# env/bin/python3 -m embedding_speed_comparisons \
#     -q "p365" \
#     --model gtr-t5-xxl \
#     --data_path data/shooting_large.json \
#     --vector_size 768 \
#     --collection "shooting_large_gtr" \
#     --log_file logs/gtr_benchmark.log \
#     -n 10 \
#     -r 3

# Qwen.
# env/bin/python3 -m embedding_speed_comparisons \
#     -q "p365" \
#     --model Qwen/Qwen3-Embedding-8B \
#     --data_path data/shooting_large.json \
#     --vector_size 4096 \
#     --collection "shooting_large_qwen" \
#     --log_file logs/qwen_benchmark.log \
#     -n 10 \
#     -r 3


# Gemini.
# env/bin/python3 -m embedding_speed_comparisons \
#     -q "p365" \
#     --model models/embedding-001 \
#     --data_path data/shooting_large.json \
#     --vector_size 768 \
#     --collection "shooting_large" \
#     --gemini \
#     --log_file logs/gemini_benchmark.log \
#     -n 10 \
#     -r 3
