## Generate SFT Data Part 1 - Best Move Prediction Reasoning

1. Download dataset from [Lichess](https://database.lichess.org/#evals) to `data/lichess_db_dval.jsonl.zst`
2. Sample the first 20,000 data without decompressing
```
zstd -cd data/lichess_db_eval.jsonl.zst | head -n 20000 > data/lichess_db_eval_head20k.jsonl
```
3. Generate SFT data
```
python generate_sft_data.py \
    --data_path "data/lichess_db_eval_head20k.jsonl" \
    --max_candidates 3 \
    --max_moves 5 \
    --template_path "reasoning_template.txt" \
    --out_path "data/lichess_eval_sft.jsonl" \
    --shuffle_candidates
```