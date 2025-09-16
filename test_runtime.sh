#!/usr/bin/env bash

# Default variables
base_model_id="meta-llama/Llama-3.1-8B-Instruct"
assistant_model_id="meta-llama/Llama-3.2-1B-Instruct"
params_file_name="ckpt/llama_3.1_8b_gsm8k_bsz4_proj.bin"
base_model_ckpt="None"
assistant_model_ckpt="None"
num_thought_tokens=4
num_return_sequences=1
task_name="gsm8k"
seed=42
test_k=0
tune_assistant_model=false
print_input=false
print_response=false
log_dir="logs"
run_name=""

# Argument parsing
while [[ $# -gt 0 ]]; do
    case $1 in
        --base_model_id) base_model_id="$2"; shift 2 ;;
        --assistant_model_id) assistant_model_id="$2"; shift 2 ;;
        --params_file_name) params_file_name="$2"; shift 2 ;;
        --assistant_model_ckpt) assistant_model_ckpt="$2"; shift 2 ;;
        --base_model_ckpt) base_model_ckpt="$2"; shift 2 ;;
        --num_thought_tokens) num_thought_tokens="$2"; shift 2 ;;
        --num_return_sequences) num_return_sequences="$2"; shift 2 ;;
        --task_name) task_name="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        --test_k) test_k="$2"; shift 2 ;;
        --tune_assistant_model) tune_assistant_model=true; shift ;;
        --print_input) print_input=true; shift ;;
        --print_response) print_response=true; shift ;;
        --log_dir) log_dir="$2"; shift 2 ;;
        --run_name) run_name="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; shift ;;
    esac
done

# Display basic configuration
echo "=================== [Default Args] ====================="
echo "Large Model ID: ${base_model_id}"
echo "Small Model ID: ${assistant_model_id}"
echo "Params File Name: ${params_file_name}"
echo "Seed: ${seed}"
echo "--------------------------------------------------------"

base_model_name="${base_model_id#*/}"
assistant_model_name="${assistant_model_id#*/}"

# Ensure log directory exists
mkdir -p "${log_dir}"

######################### GSM8K #########################
task_name="gsm8k"
start_time=$(date +%s)

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "Eval Task: ${task_name} at $(date)"

cmd="python evaluate_softcot.py \
--base_model_id \"${base_model_id}\" \
--assistant_model_id \"${assistant_model_id}\" \
--params_file_name \"${params_file_name}\" \
--assistant_model_ckpt \"${assistant_model_ckpt}\" \
--base_model_ckpt \"${base_model_ckpt}\" \
--num_thought_tokens ${num_thought_tokens} \
--num_return_sequences ${num_return_sequences} \
--task_name \"${task_name}\" \
--seed ${seed} \
--test_k ${test_k}"

log_file_name="${log_dir}/Test-time-${task_name}-${base_model_name}-tokens${num_thought_tokens}-seed${seed}.log"

# Run the command and redirect output
echo "${cmd} > \"${log_file_name}\""
eval "${cmd} > \"${log_file_name}\""

# Display the script end time
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Evaluation for dataset ${dataset} finished at: $(date)"
echo "Elapsed time: ${elapsed_time} seconds"
