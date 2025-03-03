#!/bin/bash

# Check for the correct number of arguments.
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <iterations> <segments> <starting seed>"
  exit 1
fi

iterations=$1
segments=$2
starting_seed=$3


# Loop for the specified number of iterations.
for ((i=0; i<iterations; i++)); do
  current_seed=$((starting_seed + i))
  echo "Iteration $((i+1)): running $segments segments with seed $current_seed"
  
  python infer.py \
    --cuda_idx 0 \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-cot \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt ../prompt_egs/genre.txt \
    --lyrics_txt ../prompt_egs/lyrics.txt \
    --run_n_segments $segments \
    --stage2_batch_size 5 \
    --output_dir ../output \
    --max_new_tokens 3000 \
    --repetition_penalty 1.1 \
    --seed=$current_seed
done
