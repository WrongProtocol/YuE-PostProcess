#!/bin/bash

# Check for the correct number of arguments.
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <starting_seed> <iterations>"
  exit 1
fi

starting_seed=$1
iterations=$2

# Loop for the specified number of iterations.
for ((i=0; i<iterations; i++)); do
  current_seed=$((starting_seed + i))
  echo "Iteration $((i+1)): running with seed $current_seed"
  
  python infer.py \
    --cuda_idx 0 \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-cot \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt ../prompt_egs/genre.txt \
    --lyrics_txt ../prompt_egs/lyrics.txt \
    --run_n_segments 8 \
    --stage2_batch_size 5 \
    --output_dir ../output \
    --max_new_tokens 3000 \
    --repetition_penalty 1.1 \
    --seed=$current_seed
done
