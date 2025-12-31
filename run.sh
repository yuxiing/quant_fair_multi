CACHE_DIR="/home/xxx/models"
OUTPUT_BASE_DIR="outputs_FFB/"
CUDA_DEVICES="0"

TOTAL_TASKS=120
TOTAL_PROCESSED=0

MODEL_TO_USE="Qwen/Qwen2.5-32B-Instruct"
QUANT=""
DTYPE_ARG="bfloat16"

LANGUAGES=("en" "cn" "es" "nl" "tr" "jp" "kr" "fr" "basq" "urdu" "catalan" "ln" )
declare -A LANG_CATEGORIES
LANG_CATEGORIES["en"]="Age Appearance Education Gender Health LGBTQ Nationality Race Religion SES"
LANG_CATEGORIES["cn"]="Age Appearance Education Gender Health LGBTQ Nationality Race Religion SES"
LANG_CATEGORIES["es"]="Age Appearance Education Gender Health LGBTQ Nationality Race Religion SES"
LANG_CATEGORIES["nl"]="Age Appearance Education Gender Health LGBTQ Nationality Race Religion SES"
LANG_CATEGORIES["tr"]="Age Appearance Education Gender Health LGBTQ Nationality Race Religion SES"
LANG_CATEGORIES["jp"]="Age Appearance Education Gender Health LGBTQ Nationality Race Religion SES"
LANG_CATEGORIES["kr"]="Age Appearance Education Gender Health LGBTQ Nationality Race Religion SES"
LANG_CATEGORIES["fr"]="Age Appearance Education Gender Health LGBTQ Nationality Race Religion SES"
LANG_CATEGORIES["basq"]="Age Appearance Education Gender Health LGBTQ Nationality Race Religion SES"
LANG_CATEGORIES["urdu"]="Age Appearance Education Gender Health LGBTQ Nationality Race Religion SES"
LANG_CATEGORIES["catalan"]="Age Appearance Education Gender Health LGBTQ Nationality Race Religion SES"
LANG_CATEGORIES["ln"]="Age Appearance Education Gender Health LGBTQ Nationality Race Religion SES"


export HF_HOME="$CACHE_DIR"
export TRANSFORMERS_CACHE="$CACHE_DIR"

printf "\n===============================================\n"
printf "Model: %s\n" "$MODEL_TO_USE"
printf "Languages in this part: %s\n" "${LANGUAGES[*]}"
printf "Total tasks in this part: %d\n" "$TOTAL_TASKS"
printf "Started at: $(date)\n"
printf "===============================================\n\n"

for LANG in "${LANGUAGES[@]}"; do
  echo "  -> Language: $LANG"
  IFS=' ' read -ra CATEGORIES <<< "${LANG_CATEGORIES[$LANG]}"
  for category in "${CATEGORIES[@]}"; do
    TOTAL_PROCESSED=$((TOTAL_PROCESSED + 1))
    
    MODEL_NAME_CLEAN=$(echo "$(basename "$MODEL_TO_USE")" | tr '/' '_')
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_NAME_CLEAN}/${LANG}/${category}"
    RESULTS_FILE="${OUTPUT_DIR}/ffb_results.jsonl"
    
    if [ -f "$RESULTS_FILE" ]; then
      printf "    -> [%3d/%d] [SKIPPED] %s (results exist)\n" "$TOTAL_PROCESSED" "$TOTAL_TASKS" "$category"
      continue
    fi
    
    printf "    -> [%3d/%d] [RUNNING] %s\n" "$TOTAL_PROCESSED" "$TOTAL_TASKS" "$category"
    
    PYTHON_CMD="CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python vanilla_f2bench.py \
      --model_name_or_path \"$MODEL_TO_USE\" \
      --dtype \"$DTYPE_ARG\" \
      --language \"$LANG\" \
      --max_generated_tokens 1500 \
      --output_path \"$OUTPUT_BASE_DIR\" \
      --batch_size 500 \
      --category \"$category\" \
      --cache_dir \"$CACHE_DIR\""
    
    if [ -n "$QUANT" ]; then
      PYTHON_CMD="$PYTHON_CMD --quantization \"$QUANT\""
    fi
    
    eval $PYTHON_CMD
    
    if [ $? -eq 0 ] && [ -f "$RESULTS_FILE" ]; then
      printf "    -> ✓ SUCCESS: %s\n" "$category"
    elif [ $? -eq 0 ]; then
      printf "    -> ⚠ WARNING: %s (script succeeded but no results file found)\n" "$category"
    else
      printf "    -> ✗ FAILED: %s (error code $?)\n" "$category"
    fi
  done
done
