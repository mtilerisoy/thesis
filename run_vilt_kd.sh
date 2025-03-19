#!/bin/bash
# filepath: /home/mileriso/thesis/custom_qat_run_multiple.sh

# Default values for output and error files (literal paths with filenames and extensions)
gpu_param=1
lr=0.00001
alpha_kd=0.5
kd_layer=0

# Parse arguments for --lr, --alpha_kd, --kd_layer and --gpu parameter.
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --lr)
            lr="$2"
            shift # past argument
            shift # past value
            ;;
        --alpha_kd)
            alpha_kd="$2"
            shift
            shift
            ;;
        --kd_layer)
            kd_layer="$2"
            shift
            shift
            ;;
        --gpu)
            gpu="$2"
            shift
            shift
            ;;
        *)    # Ignore unrecognized parameters.
            shift
            ;;
    esac
done

# If no Learning Rate argument was provided, default to 0.000001.
if [ -z "${lr}" ]; then
    lr=0.00001
fi

# If no Alpha KD argument was provided, default to 0.5
if [ -z "${alpha_kd}" ]; then
    alpha_kd=0.5
fi

if [ -z "${kd_layer}" ]; then
    kd_layer=0
fi

# If no GPU argument was provided, default to 1.
if [ -z "${gpu}" ]; then
    gpu=1
fi

exp_folder="./experiments/logs"
output_file="${exp_folder}/output.txt"
error_file="${exp_folder}/error.txt"

mkdir -p "${exp_folder}"

touch "${output_file}"
touch "${error_file}"

echo "Running with Learning Rate: ${lr}"
echo "Running with Alpha KD: ${alpha_kd}"
echo "Running with KD Layer: ${kd_layer}"
echo "Running with GPU: ${gpu_param}"
echo "Output file: ${output_file}"
echo "Error file: ${error_file}"

# Clear previous output and error files.
> "${output_file}"
> "${error_file}"

python run_vilt_kd.py   --epochs None \
                        --max_steps 1000 \
                        --learning_rate ${lr} \
                        --dataset "nlvr2_original" \
                        --alpha_kd ${alpha_kd} \
                        --kd_layer ${kd_layer} \
                        --log_dir ${exp_folder} >> "${output_file}" 2>> "${error_file}"
