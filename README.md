# vectors-deployable-speedtest

For benchmarking embedding models.

## HuggingFace models

You can download directly into directory: `models/`

Then you can refer using `--model`:

```bash
env/bin/python3 -m embedding_speed_comparisons \
  -q "p365" \
  --model "./models/gtr-t5-xxl" \
  --data_path data/shooting_large.json \
  --vector_size 768 \
  --collection "shooting_large_gtr" \
  --log logs/gtr_benchmark.log \
  -n 10 \
  -r 3
```

## Common pitfalls

**401 error when pulling from HuggingFace**

Make sure you are not using an API key like `HF_TOKEN`.
