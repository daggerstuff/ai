# Edge Case Dialogue Generation Scripts

This folder contains scripts and data for generating realistic therapy session dialogues based on edge case scenarios. These dialogues can be used for training, research, and testing purposes.

## Overview

The scripts in this package allow you to:

1. Generate dialogue outputs from edge case prompts using the Ollama LLM interface
2. Validate that generated dialogues meet required formatting and content criteria
3. Create reports on dialogue quality and format compliance

## Requirements

- Node.js (v22 preferred)
- [Ollama](https://ollama.ai/) installed and accessible in your PATH
- The `artifish/llama3.2-uncensored` model pulled in Ollama

## Quick Start

The easiest way to run the scripts is to use the provided npm scripts:

```bash
# Generate individual dialogues interactively
pnpm run generate-dialogues

# Generate multiple dialogues in batch mode
pnpm run batch-generate-dialogues

# Validate generated dialogues
pnpm run validate-dialogues

# Run the full pipeline (interactive menu)
pnpm run dialogue-pipeline
```

## Scripts

The following scripts are available in the `src/scripts` directory:

### `generate_dialogues.js`

An interactive script that allows you to select and generate dialogues one at a time:

- Displays a list of available prompts
- Lets you choose which prompt to generate a dialogue for
- Runs the LLM to create the dialogue
- Saves the output to a file in the `ai/generated_dialogues` directory

Usage:
```bash
node src/scripts/generate_dialogues.js
# or
pnpm run generate-dialogues
```

### `batch_generate_dialogues.js`

Processes multiple prompts automatically, useful for generating all dialogues at once:

- Processes all prompts from the JSONL file, or a specified range
- Supports concurrent processing of multiple prompts
- Skips prompts that already have generated outputs
- Creates a summary report of the generation process

Usage:
```bash
node src/scripts/batch_generate_dialogues.js
# or
pnpm run batch-generate-dialogues
```

When running the batch script, you'll be prompted for:
- **Concurrency**: How many prompts to process simultaneously (1-4)
- **Start index**: Which prompt to start from (0-based)
- **Maximum prompts**: How many prompts to process

### `validate_dialogues.js`

Checks the generated dialogues against quality criteria:

- Verifies that dialogues have the required number of turns
- Checks that turns alternate properly between Therapist and Client
- Looks for internal monologue, non-verbal cues, and physical symptoms
- Ensures ethical dilemmas are mentioned
- Validates that the outcome matches expectations (especially for "must fail" scenarios)
- Generates a detailed validation report in Markdown format

Usage:
```bash
node src/scripts/validate_dialogues.js
# or
pnpm run validate-dialogues
```

### `run_full_dialogue_pipeline.js`

A master script that provides an interactive menu to run any of the above scripts:

- Offers options to run individual dialogue generation, batch generation, or validation
- Includes a "full pipeline" option that runs batch generation followed by validation
- Provides a user-friendly interface for all dialogue generation tasks

Usage:
```bash
node src/scripts/run_full_dialogue_pipeline.js
# or
pnpm run dialogue-pipeline
```

## Output Files

The scripts generate the following output files in the `ai/generated_dialogues` directory:

- **Dialogue files**: Named `edge-XXX_scenario-type.txt`, containing the generated dialogues
- **Validation report**: `validation_report.md`, detailing which dialogues passed or failed validation
- **Batch summary**: `batch_generation_summary.md`, summarizing the results of batch generation

## Edge Case Prompts File

The dialogues are generated based on edge case prompts defined in the `ai/edge_case_prompts.jsonl` file. Each line in this file is a separate JSON object with the following structure:

```json
{
  "prompt_id": "edge-001", 
  "scenario_type": "therapist_failure", 
  "fail_no_matter_what": false,
  "instructions": "Detailed instructions for the LLM..."
}
```

The `instructions` field contains detailed instructions for the LLM to generate the dialogue.

## Customization

To customize the generation process:

1. **Add new prompts**: Add new JSON objects to the `edge_case_prompts.jsonl` file
2. **Change LLM model**: Edit the `MODEL` constant in the scripts to use a different Ollama model
3. **Adjust temperature**: Modify the `TEMPERATURE` setting to control output randomness
4. **Change validation criteria**: Edit the constants in `validate_dialogues.js` to adjust validation rules

## Troubleshooting

- **Ollama not found**: Ensure Ollama is installed and in your PATH
- **Model not available**: The script will attempt to pull the model if it's not found
- **Empty or error responses**: Try lowering the temperature or check if Ollama is responding correctly
- **Low validation scores**: Review the validation report for specific issues and adjust the prompts as needed

## Notes

- All scripts include error handling and will create detailed log files
- The validation script provides specific feedback on which aspects of each dialogue need improvement
- You can re-run validation independently after fixing issues in the dialogues

## For Development

If you need to modify or extend these scripts:

1. The dialogue file naming convention is important for validation: `{prompt_id}_{scenario_type}.txt`
2. The `scenarioType` and `fail_no_matter_what` fields are used to determine validation criteria
3. You can add additional patterns to check in the validation script's pattern arrays 