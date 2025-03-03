#!/bin/bash
# filepath: /home/mileriso/thesis/custom_qat_run_multiple.sh

# Default values for output and error files (literal paths with filenames and extensions)
output_file="./output_inference_results.txt"
error_file="./error_inference_results.txt"

echo "Output file: ${output_file}"
echo "Error file: ${error_file}"

# Clear previous output and error files.
> "${output_file}"
> "${error_file}"

# Loop over a series of parameter sets.
for dataset in nlvr2_ood nlvr2_original; do
  for model in "vilt" "meter"; do
    echo "┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐" >> "${output_file}"
    echo "│                                                                                                     │" >> "${output_file}"
    echo "│  Running with: dataset=${dataset}, model=${model}                                                   │" >> "${output_file}"
    echo "│                                                                                                     │" >> "${output_file}"
    echo "└─────────────────────────────────────────────────────────────────────────────────────────────────────┘" >> "${output_file}"
    python custom_qat_inference.py --dataset "${dataset}" \
                                  --model "${model}" >> "${output_file}" 2>> "${error_file}"
  done
done