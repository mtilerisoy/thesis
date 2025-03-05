#!/bin/bash
# filepath: /home/mileriso/thesis/custom_qat_run_multiple.sh

# Default values for output and error files (literal paths with filenames and extensions)
output_file="./output.txt"
error_file="./error.txt"
gpu_param=""

# Parse arguments for --output_file, --error_file, and --gpu parameter.
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --output_file)
            output_file="$2"
            shift # past argument
            shift # past value
            ;;
        --error_file)
            error_file="$2"
            shift
            shift
            ;;
        --gpu)
            gpu_param="$2"
            shift
            shift
            ;;
        *)    # Ignore unrecognized parameters.
            shift
            ;;
    esac
done

# If no GPU argument was provided, default to 1.
if [ -z "${gpu_param}" ]; then
    gpu_param=1
fi

echo "Running with GPU: ${gpu_param}"
echo "Output file: ${output_file}"
echo "Error file: ${error_file}"

# Clear previous output and error files.
> "${output_file}"
> "${error_file}"

# Loop over a series of parameter sets.
for dataset in nlvr2_original nlvr2_ood; do
  for percentage in 1 0.5; do
    for lr in 0.01 0.001 0.0001 0.00001; do
      for epochs in 3 10 20; do
        echo "┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐" >> "${output_file}"
        echo "│                                                                                                     │" >> "${output_file}"
        echo "│  Running with: epochs=${epochs}, learning_rate=${lr}, dataset=${dataset}, percentage=${percentage}                        │" >> "${output_file}"
        echo "│                                                                                                     │" >> "${output_file}"
        echo "└─────────────────────────────────────────────────────────────────────────────────────────────────────┘" >> "${output_file}"
        python run_meter_kd.py --epochs ${epochs} \
                              --learning_rate ${lr} \
                              --dataset "${dataset}" \
                              --percentage ${percentage} \
                              --gpu ${gpu_param} >> "${output_file}" 2>> "${error_file}"
        echo "----------------------------------------------------" >> "${output_file}"
      done
    done
  done
done