#!/bin/bash

# Define the parameters for the script
val_schemes=( "three_blocks" "last_block" "ks_no_summer" "ks_alg" "stratified_small" "stratified")
seeds=(2 12 22 32 42)
test_seasons=(2015 2016 2017 2018)
learning_rates=(0.001 0.00001)
batch_sizes=(7 14 28)
dropout_probs=(0)
hidden_layers=(2)
hidden_size=(128 256 512)
window_sizes=(14)
num_queries=(60 80 100)
forecast_horizon=(0)

# Define the timestamp and set_name
timestamp=$(date +%Y%m%d_%H%M%S)
set_name="experiment_set_${timestamp}"

# Define the content for the readme file
readme_content="This directory contains the results of the experiments conducted on ${timestamp}. with minmax normalisation,  patience set to 20, filter-out sparse entries before smoothing/semantic filter, queries with contagious period seed, top 1000 queries, with past 4 years' correlation analysis, fix best model selection, using the same pivot df for validation selections, for different seeds, change to QY's setting"

# Create experiment directory
create_experiment_dir() {
    root_dir=$1
    experiment_name=$2
    mkdir -p "${root_dir}/${experiment_name}"
    
    # Copy experiment.py and test.sh into the experiment directory
    cp ./src/experiment.py "${root_dir}/"
    cp ./test.sh "${root_dir}/"
}

# Create the root directory and readme.txt
root_dir="./experiments/England/${set_name}"
mkdir -p "${root_dir}"
echo "${readme_content}" > "${root_dir}/readme.txt"

# Log file for debugging
debug_log="${root_dir}/debug.log"
echo "[$(date +%Y%m%d_%H%M%S)] Debug log for experiments started at ${timestamp}" > "${debug_log}"

# Loop over each GPU and distribute the experiments
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
max_jobs=4
job_count=0
declare -a gpu_jobs
declare -a gpu_pids
for ((i=0; i<gpu_count; i++)); do
    gpu_jobs[i]=0
    gpu_pids[i]=0
done

# Function to manage job queue
manage_jobs() {
    # Wait for a job to finish if we have reached max_jobs
    if [ "${job_count}" -ge "${max_jobs}" ]; then
        wait -n
        job_count=$((job_count - 1))
        # Update GPU job counts
        for ((i=0; i<gpu_count; i++)); do
            if ! kill -0 "${gpu_pids[i]}" 2>/dev/null; then
                gpu_jobs[i]=0
                gpu_pids[i]=0
            fi
        done
    fi
}

for seed in "${seeds[@]}"; do
    for forecast_horizon in "${forecast_horizon[@]}"; do
        for val_scheme in "${val_schemes[@]}"; do
            # Debugging message for current validation scheme
            echo "[$(date +%Y%m%d_%H%M%S)] Starting validation scheme: ${val_scheme}" >> "${debug_log}"
            
            for test_season in "${test_seasons[@]}"; do
                experiment_name="${val_scheme}_test${test_season}_gamma${forecast_horizon}_seed${seed}"
                create_experiment_dir "${root_dir}" "${experiment_name}"

                log_file="${root_dir}/${experiment_name}/output.log"

                # Assign GPU based on test_season
                gpu=$((test_season - 2015))
                
                # Ensure only one job runs at a time for each GPU
                while [ "${gpu_jobs[${gpu}]}" -ge 1 ]; do
                    sleep 10
                    if ! kill -0 "${gpu_pids[${gpu}]}" 2>/dev/null; then
                        gpu_jobs[${gpu}]=0
                        gpu_pids[${gpu}]=0
                    fi
                done
                
                echo "[$(date +%Y%m%d_%H%M%S)] Starting process for validation scheme: ${val_scheme}, test season: ${test_season} on GPU: ${gpu}" >> "${debug_log}"
                CUDA_VISIBLE_DEVICES=${gpu} python -m src.experiment \
                    --seed ${seed} \
                    --test_season ${test_season} \
                    --forecast_horizon ${forecast_horizon} \
                    --validation_scheme ${val_scheme} \
                    --learning_rate "${learning_rates[@]}" \
                    --batch_size "${batch_sizes[@]}" \
                    --dropout_prob "${dropout_probs[@]}" \
                    --hidden_layers "${hidden_layers[@]}" \
                    --hidden_size "${hidden_size[@]}" \
                    --window_size "${window_sizes[@]}" \
                    --num_queries "${num_queries[@]}" \
                    --root_dir "${root_dir}/${experiment_name}" \
                    --patience 10\
                    2>&1 | tee "${log_file}" &
                
                pid=$!
                job_count=$((job_count + 1))
                gpu_jobs[${gpu}]=$((gpu_jobs[${gpu}] + 1))
                gpu_pids[${gpu}]=$pid

                # Manage the job queue
                manage_jobs
            done
        done
    done
done

# Wait for all remaining jobs to finish
wait

# Create a log file indicating all jobs are done
completion_time=$(date +"%Y%m%d_%H%M%S")
completion_log="../../Dropbox/logs/all_jobs_completed_${completion_time}.log"

mkdir -p ../../Dropbox/logs
echo "All jobs have been completed at ${completion_time}" > "${completion_log}"

echo "[$(date +%Y%m%d_%H%M%S)] All jobs have been submitted." >> "${debug_log}"
