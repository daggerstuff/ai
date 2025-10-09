#!/bin/bash

# Enhanced Dialogue Generation Interactive Script
# This script provides a user-friendly interface for creating synthetic therapy conversations

# ═══════════════════════════════════════════
# Colors and formatting
# ═══════════════════════════════════════════
BOLD='\033[1m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

# ═══════════════════════════════════════════
# Default values
# ═══════════════════════════════════════════
DEFAULT_INPUT="ai/edge_case_prompts_with_categories.jsonl"
DEFAULT_TEMPLATES="ai/template_examples.json"
DEFAULT_CHAIN_TEMPLATES="ai/chain_templates.json"
DEFAULT_MODEL="llama3"
DEFAULT_OUTPUT_DIR="ai/generated"

# ═══════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════

# Check for required software and packages
check_requirements() {
  echo -e "${BLUE}Checking requirements...${NC}"
  
  # Check for Python
  if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python is not installed${NC}"
    exit 1
  fi
  
  # Check for required Python packages
  REQUIRED_PACKAGES=("requests" "tqdm" "python-dotenv")
  MISSING_PACKAGES=()
  
  for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! python -c "import $package" &> /dev/null; then
      MISSING_PACKAGES+=("$package")
    fi
  done
  
  if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
    echo -e "${YELLOW}⚠️  Missing required packages:${NC}"
    for package in "${MISSING_PACKAGES[@]}"; do
      echo "  • $package"
    done
    
    read -p "Would you like to install these packages now? (y/n): " INSTALL_PACKAGES
    if [[ $INSTALL_PACKAGES == "y" || $INSTALL_PACKAGES == "Y" ]]; then
      pip install "${MISSING_PACKAGES[@]}"
    else
      echo -e "${RED}❌ Required packages are missing. Exiting.${NC}"
      exit 1
    fi
  fi
  
  # Check for optional packages
  OPTIONAL_PACKAGES=("slack_sdk" "vllm")
  MISSING_OPTIONAL=()
  
  for package in "${OPTIONAL_PACKAGES[@]}"; do
    if ! python -c "import $package" &> /dev/null; then
      MISSING_OPTIONAL+=("$package")
    fi
  done
  
  if [ ${#MISSING_OPTIONAL[@]} -ne 0 ]; then
    echo -e "${YELLOW}ℹ️  Optional packages not installed:${NC}"
    for package in "${MISSING_OPTIONAL[@]}"; do
      echo "  • $package"
    done
    echo -e "${GRAY}(These enable additional features but aren't required)${NC}"
  fi
  
  echo -e "${GREEN}✓ All core requirements met!${NC}"
}

# Create output directory if it doesn't exist
ensure_output_dir() {
  if [ ! -d "$1" ]; then
    echo -e "${BLUE}📁 Creating folder: $1${NC}"
    mkdir -p "$1"
  fi
}

# ═══════════════════════════════════════════
# Main Functions
# ═══════════════════════════════════════════

# Simple conversation generation
basic_generation() {
  clear
  echo -e "${BOLD}${BLUE}⭐ Simple Conversation Creator${NC}"
  echo -e "${GRAY}Creates basic therapy conversations with minimal setup${NC}"
  echo -e "═══════════════════════════════════════════"
  
  # Get input file
  read -p "📄 Prompts file (default: $DEFAULT_INPUT): " INPUT_FILE
  INPUT_FILE=${INPUT_FILE:-$DEFAULT_INPUT}
  
  # Get model name
  read -p "🤖 AI model (default: $DEFAULT_MODEL): " MODEL
  MODEL=${MODEL:-$DEFAULT_MODEL}
  
  # Get output directory
  read -p "📁 Save to folder (default: $DEFAULT_OUTPUT_DIR): " OUTPUT_DIR
  OUTPUT_DIR=${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}
  
  ensure_output_dir "$OUTPUT_DIR"
  
  # Setup output filenames
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  OUTPUT_FILE="$OUTPUT_DIR/dialogues_${TIMESTAMP}.jsonl"
  RAW_OUTPUT="$OUTPUT_DIR/raw_${TIMESTAMP}.jsonl"
  LOG_FILE="$OUTPUT_DIR/log_${TIMESTAMP}.log"
  
  echo -e "${YELLOW}🚀 Creating conversations...${NC}"
  echo -e "📄 Input: $INPUT_FILE"
  echo -e "🤖 Model: $MODEL"
  echo -e "💾 Output: $OUTPUT_FILE"
  
  python ai/generate_synthetic.py \
    --input "$INPUT_FILE" \
    --model "$MODEL" \
    --output "$OUTPUT_FILE" \
    --raw_output "$RAW_OUTPUT" \
    --log "$LOG_FILE"
  
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Success! Conversations created.${NC}"
    echo -e "💾 Output: $OUTPUT_FILE"
    echo -e "📊 Raw data: $RAW_OUTPUT"
    echo -e "📝 Log: $LOG_FILE"
  else
    echo -e "${RED}❌ Something went wrong. Check the log: $LOG_FILE${NC}"
  fi
  
  read -p "Press Enter to continue..."
}

# Using templates for better results
template_generation() {
  clear
  echo -e "${BOLD}${BLUE}🔮 Template-Based Conversation Creator${NC}"
  echo -e "${GRAY}Creates conversations using pre-designed templates for better results${NC}"
  echo -e "═══════════════════════════════════════════"
  
  # Get input file
  read -p "📄 Prompts file (default: $DEFAULT_INPUT): " INPUT_FILE
  INPUT_FILE=${INPUT_FILE:-$DEFAULT_INPUT}
  
  # Get templates file
  read -p "🧩 Templates file (default: $DEFAULT_TEMPLATES): " TEMPLATES_FILE
  TEMPLATES_FILE=${TEMPLATES_FILE:-$DEFAULT_TEMPLATES}
  
  # Get model name
  read -p "🤖 AI model (default: $DEFAULT_MODEL): " MODEL
  MODEL=${MODEL:-$DEFAULT_MODEL}
  
  # Get output directory
  read -p "📁 Save to folder (default: $DEFAULT_OUTPUT_DIR): " OUTPUT_DIR
  OUTPUT_DIR=${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}
  
  ensure_output_dir "$OUTPUT_DIR"
  
  # Setup output filenames
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  OUTPUT_FILE="$OUTPUT_DIR/template_dialogues_${TIMESTAMP}.jsonl"
  RAW_OUTPUT="$OUTPUT_DIR/raw_template_${TIMESTAMP}.jsonl"
  LOG_FILE="$OUTPUT_DIR/log_template_${TIMESTAMP}.log"
  
  echo -e "${YELLOW}🚀 Creating template-based conversations...${NC}"
  echo -e "📄 Input: $INPUT_FILE"
  echo -e "🧩 Templates: $TEMPLATES_FILE"
  echo -e "🤖 Model: $MODEL"
  echo -e "💾 Output: $OUTPUT_FILE"
  
  python ai/generate_synthetic.py \
    --input "$INPUT_FILE" \
    --templates "$TEMPLATES_FILE" \
    --model "$MODEL" \
    --output "$OUTPUT_FILE" \
    --raw_output "$RAW_OUTPUT" \
    --log "$LOG_FILE"
  
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Success! Template-based conversations created.${NC}"
    echo -e "💾 Output: $OUTPUT_FILE"
    echo -e "📊 Raw data: $RAW_OUTPUT"
    echo -e "📝 Log: $LOG_FILE"
  else
    echo -e "${RED}❌ Something went wrong. Check the log: $LOG_FILE${NC}"
  fi
  
  read -p "Press Enter to continue..."
}

# Advanced multi-step generation
chain_generation() {
  clear
  echo -e "${BOLD}${BLUE}🔄 Advanced Multi-Step Creator${NC}"
  echo -e "${GRAY}Creates conversations with additional outputs like notes or critiques${NC}"
  echo -e "═══════════════════════════════════════════"
  
  # Get input file
  read -p "📄 Prompts file (default: $DEFAULT_INPUT): " INPUT_FILE
  INPUT_FILE=${INPUT_FILE:-$DEFAULT_INPUT}
  
  # Get templates file
  read -p "🧩 Templates file (default: $DEFAULT_TEMPLATES): " TEMPLATES_FILE
  TEMPLATES_FILE=${TEMPLATES_FILE:-$DEFAULT_TEMPLATES}
  
  # Get chain templates file
  read -p "⛓️ Chain templates file (default: $DEFAULT_CHAIN_TEMPLATES): " CHAIN_TEMPLATES_FILE
  CHAIN_TEMPLATES_FILE=${CHAIN_TEMPLATES_FILE:-$DEFAULT_CHAIN_TEMPLATES}
  
  # Get chain type
  echo -e "${CYAN}What additional content would you like to generate?${NC}"
  echo -e " 1) 👨‍⚕️ Supervisor feedback"
  echo -e " 2) 📝 Session notes"
  echo -e " 3) 🔍 Ethical analysis"
  echo -e " 4) 📋 Treatment plan"
  read -p "Select (1-4, default: 1): " CHAIN_TYPE_OPTION
  
  case $CHAIN_TYPE_OPTION in
    2) CHAIN_TYPE="session_note" ;;
    3) CHAIN_TYPE="ethical_analysis" ;;
    4) CHAIN_TYPE="treatment_plan" ;;
    *) CHAIN_TYPE="supervisor_critique" ;;
  esac
  
  # Get model name
  read -p "🤖 AI model (default: $DEFAULT_MODEL): " MODEL
  MODEL=${MODEL:-$DEFAULT_MODEL}
  
  # Get output directory
  read -p "📁 Save to folder (default: $DEFAULT_OUTPUT_DIR): " OUTPUT_DIR
  OUTPUT_DIR=${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}
  
  ensure_output_dir "$OUTPUT_DIR"
  
  # Setup output filenames
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  OUTPUT_FILE="$OUTPUT_DIR/chain_dialogues_${TIMESTAMP}.jsonl"
  RAW_OUTPUT="$OUTPUT_DIR/raw_chain_${TIMESTAMP}.jsonl"
  LOG_FILE="$OUTPUT_DIR/log_chain_${TIMESTAMP}.log"
  
  echo -e "${YELLOW}🚀 Creating advanced multi-step content...${NC}"
  echo -e "📄 Input: $INPUT_FILE"
  echo -e "🧩 Templates: $TEMPLATES_FILE"
  echo -e "⛓️ Chain Templates: $CHAIN_TEMPLATES_FILE"
  echo -e "🔄 Additional content: $CHAIN_TYPE"
  echo -e "🤖 Model: $MODEL"
  echo -e "💾 Output: $OUTPUT_FILE"
  
  python ai/generate_synthetic.py \
    --input "$INPUT_FILE" \
    --templates "$TEMPLATES_FILE" \
    --chain \
    --chain_type "$CHAIN_TYPE" \
    --chain_templates "$CHAIN_TEMPLATES_FILE" \
    --model "$MODEL" \
    --output "$OUTPUT_FILE" \
    --raw_output "$RAW_OUTPUT" \
    --log "$LOG_FILE"
  
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Success! Advanced content created.${NC}"
    echo -e "💾 Output: $OUTPUT_FILE"
    echo -e "📊 Raw data: $RAW_OUTPUT"
    echo -e "📝 Log: $LOG_FILE"
  else
    echo -e "${RED}❌ Something went wrong. Check the log: $LOG_FILE${NC}"
  fi
  
  read -p "Press Enter to continue..."
}

# Using external AI APIs
api_generation() {
  clear
  echo -e "${BOLD}${BLUE}🌐 External API Creator${NC}"
  echo -e "${GRAY}Uses commercial APIs like OpenAI or Together.ai for better quality${NC}"
  echo -e "═══════════════════════════════════════════"
  
  # Get input file
  read -p "📄 Prompts file (default: $DEFAULT_INPUT): " INPUT_FILE
  INPUT_FILE=${INPUT_FILE:-$DEFAULT_INPUT}
  
  # Get templates file
  read -p "🧩 Templates file (default: $DEFAULT_TEMPLATES): " TEMPLATES_FILE
  TEMPLATES_FILE=${TEMPLATES_FILE:-$DEFAULT_TEMPLATES}
  
  # Get API Key
  read -p "🔑 API Key: " API_KEY
  if [ -z "$API_KEY" ]; then
    echo -e "${RED}❌ API Key is required${NC}"
    read -p "Press Enter to continue..."
    return
  fi
  
  # Get API URL
  read -p "🔗 API URL (example: https://api.together.xyz/v1/completions): " API_URL
  if [ -z "$API_URL" ]; then
    echo -e "${RED}❌ API URL is required${NC}"
    read -p "Press Enter to continue..."
    return
  fi
  
  # Get model name
  read -p "🤖 Model name (example: llama-3-8b-instruct): " MODEL
  if [ -z "$MODEL" ]; then
    echo -e "${RED}❌ Model name is required${NC}"
    read -p "Press Enter to continue..."
    return
  fi
  
  # Get output directory
  read -p "📁 Save to folder (default: $DEFAULT_OUTPUT_DIR): " OUTPUT_DIR
  OUTPUT_DIR=${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}
  
  ensure_output_dir "$OUTPUT_DIR"
  
  # Setup output filenames
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  OUTPUT_FILE="$OUTPUT_DIR/api_dialogues_${TIMESTAMP}.jsonl"
  RAW_OUTPUT="$OUTPUT_DIR/raw_api_${TIMESTAMP}.jsonl"
  LOG_FILE="$OUTPUT_DIR/log_api_${TIMESTAMP}.log"
  
  echo -e "${YELLOW}🚀 Creating API-powered conversations...${NC}"
  echo -e "📄 Input: $INPUT_FILE"
  echo -e "🧩 Templates: $TEMPLATES_FILE"
  echo -e "🔗 API URL: $API_URL"
  echo -e "🤖 Model: $MODEL"
  echo -e "💾 Output: $OUTPUT_FILE"
  
  python ai/generate_synthetic.py \
    --input "$INPUT_FILE" \
    --templates "$TEMPLATES_FILE" \
    --api_key "$API_KEY" \
    --api_url "$API_URL" \
    --model "$MODEL" \
    --output "$OUTPUT_FILE" \
    --raw_output "$RAW_OUTPUT" \
    --log "$LOG_FILE"
  
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Success! API-powered conversations created.${NC}"
    echo -e "💾 Output: $OUTPUT_FILE"
    echo -e "📊 Raw data: $RAW_OUTPUT"
    echo -e "📝 Log: $LOG_FILE"
  else
    echo -e "${RED}❌ Something went wrong. Check the log: $LOG_FILE${NC}"
  fi
  
  read -p "Press Enter to continue..."
}

# GPU-accelerated generation
vllm_generation() {
  clear
  echo -e "${BOLD}${BLUE}⚡ GPU-Accelerated Creator${NC}"
  echo -e "${GRAY}Uses GPU acceleration (vLLM) for much faster generation${NC}"
  echo -e "═══════════════════════════════════════════"
  
  # Check if vLLM is installed
  if ! python -c "import vllm" &> /dev/null; then
    echo -e "${RED}❌ vLLM is not installed${NC}"
    echo -e "${YELLOW}To install: ${NC}pip install vllm"
    echo -e "${GRAY}Note: vLLM requires a CUDA-compatible GPU${NC}"
    read -p "Press Enter to continue..."
    return
  fi
  
  # Get input file
  read -p "📄 Prompts file (default: $DEFAULT_INPUT): " INPUT_FILE
  INPUT_FILE=${INPUT_FILE:-$DEFAULT_INPUT}
  
  # Get templates file
  read -p "🧩 Templates file (default: $DEFAULT_TEMPLATES): " TEMPLATES_FILE
  TEMPLATES_FILE=${TEMPLATES_FILE:-$DEFAULT_TEMPLATES}
  
  # Get model name
  read -p "🤖 Model path (example: meta-llama/Llama-3-8B): " MODEL
  if [ -z "$MODEL" ]; then
    echo -e "${RED}❌ Model path is required${NC}"
    read -p "Press Enter to continue..."
    return
  fi
  
  # Get output directory
  read -p "📁 Save to folder (default: $DEFAULT_OUTPUT_DIR): " OUTPUT_DIR
  OUTPUT_DIR=${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}
  
  ensure_output_dir "$OUTPUT_DIR"
  
  # Setup output filenames
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  OUTPUT_FILE="$OUTPUT_DIR/vllm_dialogues_${TIMESTAMP}.jsonl"
  RAW_OUTPUT="$OUTPUT_DIR/raw_vllm_${TIMESTAMP}.jsonl"
  LOG_FILE="$OUTPUT_DIR/log_vllm_${TIMESTAMP}.log"
  
  echo -e "${YELLOW}🚀 Creating GPU-accelerated conversations...${NC}"
  echo -e "📄 Input: $INPUT_FILE"
  echo -e "🧩 Templates: $TEMPLATES_FILE"
  echo -e "🤖 Model: $MODEL"
  echo -e "💾 Output: $OUTPUT_FILE"
  
  python ai/generate_synthetic.py \
    --input "$INPUT_FILE" \
    --templates "$TEMPLATES_FILE" \
    --use_vllm \
    --model "$MODEL" \
    --output "$OUTPUT_FILE" \
    --raw_output "$RAW_OUTPUT" \
    --log "$LOG_FILE"
  
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Success! GPU-accelerated conversations created.${NC}"
    echo -e "💾 Output: $OUTPUT_FILE"
    echo -e "📊 Raw data: $RAW_OUTPUT"
    echo -e "📝 Log: $LOG_FILE"
  else
    echo -e "${RED}❌ Something went wrong. Check the log: $LOG_FILE${NC}"
  fi
  
  read -p "Press Enter to continue..."
}

# Create new therapy scenarios
prompt_generation() {
  clear
  echo -e "${BOLD}${BLUE}🎭 Scenario Creator${NC}"
  echo -e "${GRAY}Creates new therapy scenarios for conversation generation${NC}"
  echo -e "═══════════════════════════════════════════"
  
  # Show available scenario types
  echo -e "${CYAN}Some available scenario types:${NC}"
  echo -e " • 🚨 suicidality - Suicide risk assessment"
  echo -e " • 💔 trauma - Trauma processing"
  echo -e " • 🛡️ boundaries - Professional boundaries"
  echo -e " • 🆘 abuse - Responding to abuse disclosures"
  echo -e " • 🤯 sadistic_client - Client with sadistic tendencies"
  echo -e " • 😈 evil_client - Client with disturbing intentions"
  echo -e " • (many more available)"
  echo -e ""
  
  # Ask if user wants to see full list
  read -p "See full list of scenario types? (y/n, default: n): " SHOW_LIST
  if [[ $SHOW_LIST == "y" || $SHOW_LIST == "Y" ]]; then
    python ai/generate_prompts.py --list-types
    echo ""
  fi
  
  # Get number of prompts
  read -p "📊 Number of scenarios to create (default: 10): " NUM_PROMPTS
  NUM_PROMPTS=${NUM_PROMPTS:-10}
  
  # Get specific scenario types
  read -p "🔍 Specific scenario types (comma-separated, empty for all): " SCENARIO_TYPES
  
  # Get dark mode percentage
  read -p "🌑 Dark mode percentage (0.0-1.0, default: 0.2): " DARK_MODE_PERCENT
  DARK_MODE_PERCENT=${DARK_MODE_PERCENT:-0.2}
  
  # Get unsavable percentage
  read -p "⚠️ Unsavable scenario percentage (0.0-1.0, default: 0.05): " UNSAVABLE_PERCENT
  UNSAVABLE_PERCENT=${UNSAVABLE_PERCENT:-0.05}
  
  # Ask if user wants edge cases only
  read -p "🔞 Generate only edge case scenarios? (y/n, default: n): " EDGE_CASES_ONLY
  EDGE_CASES_FLAG=""
  if [[ $EDGE_CASES_ONLY == "y" || $EDGE_CASES_ONLY == "Y" ]]; then
    EDGE_CASES_FLAG="--edge-cases-only"
    
    # Recommend higher dark mode percentage for edge cases
    if (( $(echo "$DARK_MODE_PERCENT < 0.5" | bc -l) )); then
      echo -e "${YELLOW}⚠️ Tip: Consider increasing dark mode percentage for edge cases${NC}"
      read -p "Increase dark mode to 0.8? (y/n, default: y): " INCREASE_DARK
      if [[ $INCREASE_DARK != "n" && $INCREASE_DARK != "N" ]]; then
        DARK_MODE_PERCENT=0.8
        echo -e "${CYAN}Dark mode percentage set to 0.8${NC}"
      fi
    fi
  fi
  
  # Get output directory
  read -p "📁 Save to folder (default: $DEFAULT_OUTPUT_DIR): " OUTPUT_DIR
  OUTPUT_DIR=${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}
  
  ensure_output_dir "$OUTPUT_DIR"
  
  # Setup output filename
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  OUTPUT_FILE="$OUTPUT_DIR/prompts_${TIMESTAMP}.jsonl"
  
  # Build command
  CMD="python ai/generate_prompts.py --num $NUM_PROMPTS --output $OUTPUT_FILE --dark-mode-percent $DARK_MODE_PERCENT --unsavable-percent $UNSAVABLE_PERCENT $EDGE_CASES_FLAG"
  
  # Add scenario types if specified
  if [ ! -z "$SCENARIO_TYPES" ]; then
    # Convert comma-separated list to space-separated for command line argument
    SCENARIO_TYPES_ARG=$(echo $SCENARIO_TYPES | tr ',' ' ')
    CMD="$CMD --scenario-types $SCENARIO_TYPES_ARG"
  fi
  
  echo -e "${YELLOW}🚀 Creating scenarios...${NC}"
  eval $CMD
  
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Success! Scenarios created.${NC}"
    echo -e "💾 Saved to: $OUTPUT_FILE"
    
    # Show statistics about generated prompts
    echo -e "${CYAN}Prompt Statistics:${NC}"
    DARK_COUNT=$(grep -c "\"dark_mode\": true" "$OUTPUT_FILE")
    UNSAVABLE_COUNT=$(grep -c "\"unsavable\": true" "$OUTPUT_FILE")
    EDGE_CASE_COUNT=$(grep -E "\"scenario_type\": \"(sadistic_client|evil_client|therapist_failure|child_abuse|manipulative_client|sexual_abuse|domestic_violence|sexual_client|uncontrollable_escalation|intense_edge_case|counter_transference_crisis|ethical_impossibility|violent_client|forensic_nightmare|homicidal_client|mass_casualty_planning|bizarre_delusions|dangerous_paranoia|cult_deprogramming|stalker_client|active_psychosis|torture_disclosure|extreme_dissociation|boundary_dissolution|therapist_endangerment)\"" "$OUTPUT_FILE" | wc -l)
    
    echo -e " • Dark mode prompts: $DARK_COUNT ($(echo "scale=1; $DARK_COUNT*100/$NUM_PROMPTS" | bc)%)"
    echo -e " • Unsavable scenarios: $UNSAVABLE_COUNT ($(echo "scale=1; $UNSAVABLE_COUNT*100/$NUM_PROMPTS" | bc)%)"
    echo -e " • Edge case scenarios: $EDGE_CASE_COUNT ($(echo "scale=1; $EDGE_CASE_COUNT*100/$NUM_PROMPTS" | bc)%)"
    
    # Ask if user wants to use these prompts for dialogue generation
    read -p "Generate conversations from these scenarios now? (y/n, default: n): " GENERATE_DIALOGUES
    if [[ $GENERATE_DIALOGUES == "y" || $GENERATE_DIALOGUES == "Y" ]]; then
      # Get generation method
      echo -e "${CYAN}How would you like to generate conversations?${NC}"
      echo -e " 1) ⭐ Simple generation"
      echo -e " 2) 🔮 Template-based generation"
      echo -e " 3) 🔄 Advanced multi-step generation"
      echo -e " 4) 🔙 Return to main menu"
      read -p "Select (1-4, default: 2): " GEN_METHOD
      
      case $GEN_METHOD in
        1)
          # Use the generated prompts file as input for basic generation
          DEFAULT_INPUT=$OUTPUT_FILE
          basic_generation
          ;;
        3)
          # Use the generated prompts file as input for chain generation
          DEFAULT_INPUT=$OUTPUT_FILE
          chain_generation
          ;;
        4)
          # Return to main menu
          return
          ;;
        *)
          # Use the generated prompts file as input for template-based generation (default)
          DEFAULT_INPUT=$OUTPUT_FILE
          template_generation
          ;;
      esac
    fi
  else
    echo -e "${RED}❌ Scenario creation failed.${NC}"
  fi
  
  read -p "Press Enter to continue..."
}

# Run the complete process in one go
complete_pipeline() {
  clear
  echo -e "${BOLD}${BLUE}🔄 All-in-One Pipeline${NC}"
  echo -e "${GRAY}Creates scenarios and conversations in one complete workflow${NC}"
  echo -e "═══════════════════════════════════════════"
  
  # Get number of prompts
  read -p "📊 Number of scenarios to create (default: 10): " NUM_PROMPTS
  NUM_PROMPTS=${NUM_PROMPTS:-10}
  
  # Get specific scenario types
  read -p "🔍 Specific scenario types (comma-separated, empty for all): " SCENARIO_TYPES
  
  # Get dark mode percentage
  read -p "🌑 Dark mode percentage (0.0-1.0, default: 0.2): " DARK_MODE_PERCENT
  DARK_MODE_PERCENT=${DARK_MODE_PERCENT:-0.2}
  
  # Get unsavable percentage
  read -p "⚠️ Unsavable scenario percentage (0.0-1.0, default: 0.05): " UNSAVABLE_PERCENT
  UNSAVABLE_PERCENT=${UNSAVABLE_PERCENT:-0.05}
  
  # Ask if user wants edge cases only
  read -p "🔞 Generate only edge case scenarios? (y/n, default: n): " EDGE_CASES_ONLY
  EDGE_CASES_FLAG=""
  if [[ $EDGE_CASES_ONLY == "y" || $EDGE_CASES_ONLY == "Y" ]]; then
    EDGE_CASES_FLAG="--edge-cases-only"
    
    # Recommend higher dark mode percentage for edge cases
    if (( $(echo "$DARK_MODE_PERCENT < 0.5" | bc -l) )); then
      echo -e "${YELLOW}⚠️ Tip: Consider increasing dark mode percentage for edge cases${NC}"
      read -p "Increase dark mode to 0.8? (y/n, default: y): " INCREASE_DARK
      if [[ $INCREASE_DARK != "n" && $INCREASE_DARK != "N" ]]; then
        DARK_MODE_PERCENT=0.8
        echo -e "${CYAN}Dark mode percentage set to 0.8${NC}"
      fi
    fi
  fi
  
  # Get output directory
  read -p "📁 Save to folder (default: $DEFAULT_OUTPUT_DIR): " OUTPUT_DIR
  OUTPUT_DIR=${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}
  
  ensure_output_dir "$OUTPUT_DIR"
  
  # Setup output filenames with same timestamp for all pipeline outputs
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  PROMPTS_FILE="$OUTPUT_DIR/prompts_${TIMESTAMP}.jsonl"
  DIALOGUES_FILE="$OUTPUT_DIR/dialogues_${TIMESTAMP}.jsonl"
  RAW_OUTPUT="$OUTPUT_DIR/raw_${TIMESTAMP}.jsonl"
  CHAINED_OUTPUT="$OUTPUT_DIR/chained_${TIMESTAMP}.jsonl"
  LOG_FILE="$OUTPUT_DIR/pipeline_${TIMESTAMP}.log"
  
  # Step 1: Generate prompts
  echo -e "${YELLOW}🚀 Step 1: Creating scenarios...${NC}"
  CMD="python ai/generate_prompts.py --num $NUM_PROMPTS --output $PROMPTS_FILE --dark-mode-percent $DARK_MODE_PERCENT --unsavable-percent $UNSAVABLE_PERCENT $EDGE_CASES_FLAG"
  
  # Add scenario types if specified
  if [ ! -z "$SCENARIO_TYPES" ]; then
    # Convert comma-separated list to space-separated for command line argument
    SCENARIO_TYPES_ARG=$(echo $SCENARIO_TYPES | tr ',' ' ')
    CMD="$CMD --scenario-types $SCENARIO_TYPES_ARG"
  fi
  
  eval $CMD
  
  if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Scenario creation failed. Process stopped.${NC}"
    read -p "Press Enter to continue..."
    return
  fi
  
  echo -e "${GREEN}✓ Scenarios created!${NC}"
  
  # Show statistics about generated prompts
  echo -e "${CYAN}Prompt Statistics:${NC}"
  DARK_COUNT=$(grep -c "\"dark_mode\": true" "$PROMPTS_FILE")
  UNSAVABLE_COUNT=$(grep -c "\"unsavable\": true" "$PROMPTS_FILE")
  EDGE_CASE_COUNT=$(grep -E "\"scenario_type\": \"(sadistic_client|evil_client|therapist_failure|child_abuse|manipulative_client|sexual_abuse|domestic_violence|sexual_client|uncontrollable_escalation|intense_edge_case|counter_transference_crisis|ethical_impossibility|violent_client|forensic_nightmare|homicidal_client|mass_casualty_planning|bizarre_delusions|dangerous_paranoia|cult_deprogramming|stalker_client|active_psychosis|torture_disclosure|extreme_dissociation|boundary_dissolution|therapist_endangerment)\"" "$PROMPTS_FILE" | wc -l)
  
  echo -e " • Dark mode prompts: $DARK_COUNT ($(echo "scale=1; $DARK_COUNT*100/$NUM_PROMPTS" | bc)%)"
  echo -e " • Unsavable scenarios: $UNSAVABLE_COUNT ($(echo "scale=1; $UNSAVABLE_COUNT*100/$NUM_PROMPTS" | bc)%)"
  echo -e " • Edge case scenarios: $EDGE_CASE_COUNT ($(echo "scale=1; $EDGE_CASE_COUNT*100/$NUM_PROMPTS" | bc)%)"
  
  # Step 2: Get generation settings
  # Get templates file
  read -p "🧩 Templates file (default: $DEFAULT_TEMPLATES): " TEMPLATES_FILE
  TEMPLATES_FILE=${TEMPLATES_FILE:-$DEFAULT_TEMPLATES}
  
  # Get model name
  read -p "🤖 AI model (default: $DEFAULT_MODEL): " MODEL
  MODEL=${MODEL:-$DEFAULT_MODEL}
  
  # Ask about chaining
  read -p "📑 Add supplementary content (critiques, notes)? (y/n, default: n): " ENABLE_CHAIN
  CHAIN_OPTION=""
  CHAIN_TYPE_OPTION=""
  CHAIN_TEMPLATES_FILE=""
  
  if [[ $ENABLE_CHAIN == "y" || $ENABLE_CHAIN == "Y" ]]; then
    CHAIN_OPTION="--chain"
    
    # Get chain type
    echo -e "${CYAN}What additional content would you like to generate?${NC}"
    echo -e " 1) 👨‍⚕️ Supervisor feedback"
    echo -e " 2) 📝 Session notes"
    echo -e " 3) 🔍 Ethical analysis"
    echo -e " 4) 📋 Treatment plan"
    read -p "Select (1-4, default: 1): " CHAIN_TYPE_NUM
    
    case $CHAIN_TYPE_NUM in
      2) CHAIN_TYPE="session_note" ;;
      3) CHAIN_TYPE="ethical_analysis" ;;
      4) CHAIN_TYPE="treatment_plan" ;;
      *) CHAIN_TYPE="supervisor_critique" ;;
    esac
    
    CHAIN_TYPE_OPTION="--chain_type $CHAIN_TYPE"
    
    # Get chain templates file
    read -p "⛓️ Chain templates file (default: $DEFAULT_CHAIN_TEMPLATES): " CHAIN_TEMPLATES_FILE_INPUT
    CHAIN_TEMPLATES_FILE="--chain_templates ${CHAIN_TEMPLATES_FILE_INPUT:-$DEFAULT_CHAIN_TEMPLATES}"
  fi
  
  # Step 3: Generate dialogues
  echo -e "${YELLOW}🚀 Step 2: Creating conversations...${NC}"
  python ai/generate_synthetic.py \
    --input "$PROMPTS_FILE" \
    --templates "$TEMPLATES_FILE" \
    --model "$MODEL" \
    --output "$DIALOGUES_FILE" \
    --raw_output "$RAW_OUTPUT" \
    --log "$LOG_FILE" \
    $CHAIN_OPTION $CHAIN_TYPE_OPTION $CHAIN_TEMPLATES_FILE
  
  if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Conversation creation failed. See log for details.${NC}"
    read -p "Press Enter to continue..."
    return
  fi
  
  # Pipeline complete
  echo -e "${GREEN}✅ Success! Complete pipeline finished.${NC}"
  echo -e "Generated files:"
  echo -e " 📄 Scenarios: $PROMPTS_FILE"
  echo -e " 💬 Conversations: $DIALOGUES_FILE"
  echo -e " 📊 Raw data: $RAW_OUTPUT"
  if [[ $ENABLE_CHAIN == "y" || $ENABLE_CHAIN == "Y" ]]; then
    echo -e " 📑 Supplementary content: $CHAINED_OUTPUT"
  fi
  echo -e " 📝 Log: $LOG_FILE"
  
  read -p "Press Enter to continue..."
}

# Help information
show_help() {
  clear
  echo -e "${BOLD}${BLUE}ℹ️  Conversation Creator Help${NC}"
  echo -e "${GRAY}Quick guide to generating therapy conversations${NC}"
  echo -e "═══════════════════════════════════════════"
  
  echo -e "${BOLD}${CYAN}Options Explained:${NC}"
  echo -e ""
  echo -e "${BOLD}⭐ Simple Conversation Creator${NC}"
  echo -e "   Creates basic therapy conversations quickly"
  echo -e "   Good starting point for beginners"
  echo -e ""
  echo -e "${BOLD}🔮 Template-Based Conversation Creator${NC}"
  echo -e "   Uses templates to create more varied conversations"
  echo -e "   Better quality through structured prompts"
  echo -e ""
  echo -e "${BOLD}🔄 Advanced Multi-Step Creator${NC}"
  echo -e "   Creates conversations plus additional content"
  echo -e "   Adds supervisor feedback, session notes, etc."
  echo -e ""
  echo -e "${BOLD}🌐 External API Creator${NC}"
  echo -e "   Uses commercial AI services for highest quality"
  echo -e "   Requires API key and account with service"
  echo -e ""
  echo -e "${BOLD}⚡ GPU-Accelerated Creator${NC}"
  echo -e "   Much faster generation using your GPU"
  echo -e "   Requires NVIDIA GPU and vLLM package"
  echo -e ""
  echo -e "${BOLD}🎭 Scenario Creator${NC}"
  echo -e "   Creates new therapy scenarios as starting points"
  echo -e "   Categorizes by type (trauma, anxiety, etc.)"
  echo -e ""
  echo -e "${BOLD}🔄 All-in-One Pipeline${NC}"
  echo -e "   Creates everything in one streamlined process"
  echo -e "   From scenarios to final conversations"
  
  echo -e ""
  echo -e "${BOLD}${CYAN}Enhanced Prompt Generation Features:${NC}"
  echo -e ""
  echo -e "${BOLD}🌑 Dark Mode${NC}"
  echo -e "   Creates more intense, disturbing scenarios with graphic content"
  echo -e "   Useful for training therapists in extreme edge cases"
  echo -e ""
  echo -e "${BOLD}⚠️ Unsavable Scenarios${NC}"
  echo -e "   Creates scenarios designed to end badly regardless of intervention"
  echo -e "   Helps therapists practice managing clinical failures"
  echo -e ""
  echo -e "${BOLD}🔞 Edge Case Scenarios${NC}"
  echo -e "   Focused on extreme situations like sadistic clients, violence, etc."
  echo -e "   Prepares therapists for high-stakes, dangerous situations"
  
  echo -e ""
  echo -e "${BOLD}${CYAN}File Formats:${NC}"
  echo -e ""
  echo -e "${GRAY}Scenario format (JSONL):${NC}"
  echo -e '{"prompt_id": "01", "scenario_type": "trauma", "prompt": "..."}'
  echo -e ""
  echo -e "${GRAY}Template format (JSON):${NC}"
  echo -e '[{"id": "standard", "template": "Template with {scenario} placeholder"}]'
  
  read -p "Press Enter to continue..."
}

# Main menu
main_menu() {
  while true; do
    clear
    echo -e "${BOLD}${PURPLE}╔═══════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${PURPLE}║        Therapy Conversation Creator       ║${NC}"
    echo -e "${BOLD}${PURPLE}╚═══════════════════════════════════════════╝${NC}"
    echo -e ""
    echo -e "${GRAY}Create realistic therapy conversations for training and testing${NC}"
    echo -e ""
    echo -e "${BOLD}${BLUE}Choose what you'd like to do:${NC}"
    echo -e ""
    echo -e "  ${BOLD}1.${NC} ⭐ ${CYAN}Simple Conversation Creator${NC}"
    echo -e "     ${GRAY}Quick and easy therapy conversations${NC}"
    echo -e ""
    echo -e "  ${BOLD}2.${NC} 🔮 ${CYAN}Template-Based Conversation Creator${NC}"
    echo -e "     ${GRAY}Better quality with customizable templates${NC}"
    echo -e ""
    echo -e "  ${BOLD}3.${NC} 🔄 ${CYAN}Advanced Multi-Step Creator${NC}"
    echo -e "     ${GRAY}Create conversations with additional materials${NC}"
    echo -e ""
    echo -e "  ${BOLD}4.${NC} 🌐 ${CYAN}External API Creator${NC}"
    echo -e "     ${GRAY}Use services like OpenAI or Together.ai${NC}"
    echo -e ""
    echo -e "  ${BOLD}5.${NC} ⚡ ${CYAN}GPU-Accelerated Creator${NC}"
    echo -e "     ${GRAY}Much faster generation with GPU${NC}"
    echo -e ""
    echo -e "  ${BOLD}6.${NC} 🎭 ${CYAN}Scenario Creator${NC}"
    echo -e "     ${GRAY}Create new therapy scenarios${NC}"
    echo -e ""
    echo -e "  ${BOLD}7.${NC} 🔄 ${CYAN}All-in-One Pipeline${NC}"
    echo -e "     ${GRAY}Complete process from scenarios to conversations${NC}"
    echo -e ""
    echo -e "  ${BOLD}8.${NC} ℹ️  ${CYAN}Help${NC}"
    echo -e "     ${GRAY}Learn more about these tools${NC}"
    echo -e ""
    echo -e "  ${BOLD}0.${NC} 🚪 ${CYAN}Exit${NC}"
    echo -e ""
    read -p "Enter your choice (0-8): " CHOICE
    
    case $CHOICE in
      1) basic_generation ;;
      2) template_generation ;;
      3) chain_generation ;;
      4) api_generation ;;
      5) vllm_generation ;;
      6) prompt_generation ;;
      7) complete_pipeline ;;
      8) show_help ;;
      0) 
        echo -e "${GREEN}Thanks for using the Therapy Conversation Creator!${NC}"
        exit 0
        ;;
      *)
        echo -e "${RED}❌ Invalid option. Please try again.${NC}"
        sleep 1
        ;;
    esac
  done
}

# Start the program
check_requirements
ensure_output_dir "$DEFAULT_OUTPUT_DIR"
main_menu 