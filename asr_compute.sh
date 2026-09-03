#!/bin/bash

# Runs poison_attacks.py repeatedly across models, attack types and
# parameter combinations, parses the "Success: True/False" line from
# each run, and computes the Attack Success Rate (ASR) for every
# (model, attack, base->target) combination.


# a single failed python run must not kill the loop
set -uo pipefail   


# CONFIGURATIONS AND PARAMETERS
# -------------------------------------

# Models to evaluate. Comment/uncomment as needed.
MODELS=(
    "model_cat_lr0.0075"
    "model_cifar10_lr0.001_30epochs"
    "model_mnist_lr0.0075"
)

# Attacks to evaluate.
ATTACKS=(
    "fc"
    "polytope"
    "gradient"
)

# base:target class pairs per model
declare -A CLASS_PAIRS

CLASS_PAIRS["model_cat_lr0.0075"]="1:0"
CLASS_PAIRS["model_cifar10_lr0.001_30epochs"]="2:0 5:3 9:1 7:4"
CLASS_PAIRS["model_mnist_lr0.0075"]="1:7 8:3 4:9 6:5"


# How many independent runs per (model, attack, base, target) combo.
REPETITIONS=5

# Fixed attack hyperparameters (kept constant so the ASR comparison is apples-to-apples across models/attacks; change here if you want a hyperparameter sweep too, see the optional EPSILONS loop below).
POISON_NUMS=(30 60)
EPSILONS=(0.01 0.03 0.05)
STEP_SIZE=0.01
ITERATIONS=(2000 4000)
WATERMARK_OPACITYS=(0.0 0.2)
LR=0.1
EPOCHS=20

# Output locations
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="./asr_results_${TIMESTAMP}"
RAW_LOG="${OUTDIR}/raw_runs.log"
RESULTS_CSV="${OUTDIR}/results.csv"
SUMMARY_CSV="${OUTDIR}/summary.csv"

mkdir -p "$OUTDIR"
echo "model,attack,base_class,target_class,epochs,lr,poison_num,epsilon,watermark_opacity,iterations,run,success,clean_acc,poisoned_acc" > "$RESULTS_CSV"


# Extracts a specific part from the poison_attacks.py output, and returns just the value
parse_field() {
    local pattern="$1"
    local text="$2"
    echo "$text" | grep -oP "(?<=${pattern}: )[^;]+" | head -n1 | tr -d '[:space:]'
}


# MAIN LOOP
# -------------------------------------

total_runs=0
total_start=$(date +%s)

for model in "${MODELS[@]}"; do

    echo "Starting runs for model: $model - $(date +%Y-%m-%d\ %H:%M:%S)" | tee -a "$RAW_LOG"

    for attack in "${ATTACKS[@]}"; do
        pairs="${CLASS_PAIRS[$model]}"
        for pair in $pairs; do
            IFS=: read -r base target <<< "$pair"
            for eps in "${EPSILONS[@]}"; do
                for poison_num in "${POISON_NUMS[@]}"; do
                    for watermark_opacity in "${WATERMARK_OPACITYS[@]}"; do
                        for ITERATION in "${ITERATIONS[@]}"; do
                            for run in $(seq 1 "$REPETITIONS"); do
                                total_runs=$((total_runs + 1))

                                cmd="python3 poison_attacks.py -file_name $model -attack $attack -lr $LR -epochs $EPOCHS -base_class $base -target_class $target -poison_num $poison_num -epsilon $eps -step_size $STEP_SIZE -iterations $ITERATION -watermark_opacity $watermark_opacity"

                                echo "[$total_runs] - model=$model attack=$attack - Run $run - lr=$LR epochs=$EPOCHS - base=$base target=$target - num_poisoned=$poison_num epsilon=$eps step_size=$STEP_SIZE watermark_opacity=$watermark_opacity iterations=$ITERATION" | tee -a "$RAW_LOG"
                                echo "$cmd" >> "$RAW_LOG"

                                output=$(${cmd} 2>&1)
                                exit_code=$?
                                echo "$output" >> "$RAW_LOG"

                                if [ $exit_code -ne 0 ]; then
                                    echo "  -> python exited with code $exit_code, marking as FAILED run" | tee -a "$RAW_LOG"
                                    echo "${model},${attack},${base},${target},${EPOCHS},${LR},${poison_num},${eps},${watermark_opacity},${STEP_SIZE},${ITERATION},${run},ERROR,," >> "$RESULTS_CSV"
                                    continue
                                fi

                                success=$(parse_field "Success" "$output")
                                clean_acc=$(parse_field "Clean Model Accuracy" "$output")
                                poisoned_acc=$(parse_field "Poisoned Model Accuracy" "$output")

                                if [ -z "$success" ]; then
                                    echo "  -> could not parse Success field, marking as UNKNOWN" | tee -a "$RAW_LOG"
                                    success="UNKNOWN"
                                fi

                                echo "${model},${attack},${base},${target},${EPOCHS},${LR},${poison_num},${eps},${watermark_opacity},${STEP_SIZE},${ITERATION},${run},${success},${clean_acc},${poisoned_acc}" >> "$RESULTS_CSV"
                            done
                        done
                    done
                done
            done
        done
    done
done

total_end=$(date +%s)
echo "Completed $total_runs runs in $((total_end - total_start))s. Raw log: $RAW_LOG"
