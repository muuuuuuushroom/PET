#!/bin/bash
export CUDA_VISIBLE_DEVICES=4 \

resumes=(
  "vgg_baseline"
  "vgg_f4x"
  "vgg_probloss"
  "vgg_lossmixed"
)

results=()

for resume_path in "${resumes[@]}"
do
  echo "Running eval.py with resume: $resume_path"

  resume_file="outputs/WuhanMetro/${resume_path}/best_checkpoint.pth"
  vis_dir="outputs/WuhanMetro/${resume_path}/vis"

  output=$(python eval.py \
    --dataset_file="WuhanMetro" \
    --resume="$resume_file" \
    --vis_dir="$vis_dir")

  last_line=$(echo "$output" | tail -n 1)
  results+=("Result for $(basename "$resume_path"): $last_line")
done

echo
echo "===== All Results Summary ====="
for result in "${results[@]}"
do
  echo "$result"
done