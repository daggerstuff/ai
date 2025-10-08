#!/usr/bin/env node

import fs from 'fs'
import path from 'path'
import { exec, execSync } from 'child_process'
import readline from 'readline'
import { promisify } from 'util'

// Convert exec to promise-based
const execPromise = promisify(exec)

// Configuration
const MODEL = 'artifish/llama3.2-uncensored'
const INPUT_FILE = path.join(path.resolve(), 'ai/edge_case_prompts.jsonl')
const OUTPUT_DIR = path.join(path.resolve(), 'ai/generated_dialogues')
const MAX_RETRIES = 3
const TEMPERATURE = 0.8

// Ensure output directory exists
if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true })
  console.log(`Created output directory: ${OUTPUT_DIR}`)
}

// Check if Ollama is installed and the model is available
async function checkOllama() {
  try {
    // Check if Ollama is installed
    await execPromise('which ollama')
    console.log('✓ Ollama is installed')

    // Check if the model is available
    const { stdout } = await execPromise('ollama list')
    if (stdout.includes(MODEL)) {
      console.log(`✓ Model "${MODEL}" is available`)
      return true
    } else {
      console.log(`⚠ Model "${MODEL}" not found. Pulling it now...`)
      try {
        // Execute with execSync to ensure it completes before continuing
        execSync(`ollama pull ${MODEL}`, { stdio: 'inherit' })
        console.log(`✓ Successfully pulled model "${MODEL}"`)
        return true
      } catch (pullError) {
        console.error(`✗ Failed to pull model "${MODEL}": ${pullError.message}`)
        return false
      }
    }
  } catch (error) {
    console.error('✗ Ollama is not installed or not in PATH')
    console.error('Error details:', error)
    console.log('Please install Ollama: https://ollama.ai/download')
    return false
  }
}

// Read the edge case prompts from the JSONL file
function readPrompts() {
  try {
    if (!fs.existsSync(INPUT_FILE)) {
      console.error(`Input file not found: ${INPUT_FILE}`)
      return []
    }

    const data = fs.readFileSync(INPUT_FILE, 'utf8')
    return data
      .split('\n')
      .filter((line) => line.trim() !== '') // Filter out empty lines
      .map((line, index) => {
        try {
          const parsed = JSON.parse(line)
          return {
            index,
            ...parsed,
          }
        } catch (error) {
          console.error(
            `Error parsing JSON at line ${index + 1}: ${error.message}`,
          )
          return null
        }
      })
      .filter((item) => item !== null) // Filter out parsing errors
  } catch (error) {
    console.error(`Error reading input file: ${error.message}`)
    return []
  }
}

// Generate dialogue for a single prompt using Ollama
async function generateDialogue(prompt, retryCount = 0) {
  const sanitizedPrompt = prompt.instructions.replace(/"/g, '\\"')
  const ollama_cmd = `ollama run ${MODEL} "${sanitizedPrompt}" --temperature ${TEMPERATURE}`

  try {
    console.log(
      `Executing Ollama with model ${MODEL} (temperature: ${TEMPERATURE})...`,
    )
    const { stdout, stderr } = await execPromise(ollama_cmd, {
      maxBuffer: 1024 * 1024 * 10,
    }) // 10MB buffer

    if (stderr) {
      console.warn(`Ollama warning: ${stderr}`)
    }

    if (!stdout || stdout.trim() === '') {
      throw new Error('Empty response from Ollama')
    }

    return stdout
  } catch (error) {
    console.error(`Error generating dialogue: ${error.message}`)

    if (retryCount < MAX_RETRIES) {
      console.log(`Retrying (${retryCount + 1}/${MAX_RETRIES})...`)
      return generateDialogue(prompt, retryCount + 1)
    } else {
      throw new Error(
        `Failed to generate dialogue after ${MAX_RETRIES} attempts`,
      )
    }
  }
}

// Save the generated dialogue to a file
function saveDialogue(prompt, dialogue) {
  // Create a filename based on the prompt ID and scenario type
  const filename = `${prompt.prompt_id}_${prompt.scenario_type.replace(/\s+/g, '_')}.txt`
  const outputPath = path.join(OUTPUT_DIR, filename)

  // Add metadata header
  const output = `Prompt ID: ${prompt.prompt_id}
Scenario Type: ${prompt.scenario_type}
Generated with: ${MODEL}
Temperature: ${TEMPERATURE}
Date: ${new Date().toISOString()}
---

${dialogue}`

  try {
    fs.writeFileSync(outputPath, output)
    console.log(`Dialogue saved to: ${outputPath}`)
    return outputPath
  } catch (error) {
    console.error(`Error saving dialogue: ${error.message}`)
    return null
  }
}

// Create a readline interface for user interaction
function createInterface() {
  return readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  })
}

// Display a menu of available prompts and let the user select one
async function selectPrompt(prompts) {
  if (prompts.length === 0) {
    console.error('No prompts available.')
    return null
  }

  console.log('\nAvailable Prompts:')
  prompts.forEach((prompt, index) => {
    console.log(`${index + 1}. ${prompt.prompt_id} (${prompt.scenario_type})`)
  })

  const rl = createInterface()

  try {
    const answer = await new Promise((resolve) => {
      rl.question('\nSelect a prompt (number) or "q" to quit: ', resolve)
    })

    rl.close()

    if (answer.toLowerCase() === 'q') {
      return null
    }

    const selection = parseInt(answer, 10)
    if (isNaN(selection) || selection < 1 || selection > prompts.length) {
      console.error('Invalid selection. Please try again.')
      return selectPrompt(prompts)
    }

    return prompts[selection - 1]
  } catch (error) {
    console.error(`Error during prompt selection: ${error.message}`)
    rl.close()
    return null
  }
}

async function main() {
console.log('Edge Case Dialogue Generator')
console.log('===========================\n')

// Check Ollama installation
const ollamaAvailable = await checkOllama()
if (!ollamaAvailable) {
  console.error('Ollama is required to generate dialogues.')
  process.exit(1)
}

// Read the prompts
const prompts = readPrompts()
if (prompts.length === 0) {
  console.error('No prompts found in the input file.')
  process.exit(1)
}

console.log(`Found ${prompts.length} prompts in ${INPUT_FILE}`)

await promptLoop(prompts);
}

// Moved main interactive loop into separate async function to avoid await in loop
async function promptLoop(prompts) {
// Let the user select a prompt
const selectedPrompt = await selectPrompt(prompts)
if (!selectedPrompt) {
  console.log('Exiting...')
  return;
}

console.log(
  `\nSelected: ${selectedPrompt.prompt_id} (${selectedPrompt.scenario_type})`,
)
console.log('Generating dialogue...')

try {
  // Generate and save the dialogue
  const dialogue = await generateDialogue(selectedPrompt)
  const outputPath = saveDialogue(selectedPrompt, dialogue)

  if (outputPath) {
    console.log('\nDialogue generation complete!')

    // Ask if the user wants to generate another dialogue
    const answer = await askGenerateAnother();
    if (answer.toLowerCase() === 'y') {
      await promptLoop(prompts);
    } else {
      console.log('Exiting...')
    }
  }
} catch (error) {
  console.error(`Failed to generate dialogue: ${error.message}`)
}
}

/**
 * Prompt the user whether to generate another dialogue.
 * Returns the answer string.
 */
async function askGenerateAnother() {
  const rl = createInterface();
  try {
    return await new Promise((resolve) => {
      rl.question(
        '\nWould you like to generate another dialogue? (y/n): ',
        resolve,
      );
    });
  } finally {
    rl.close();
  }
}

// Run the main function
main().catch((error) => {
  console.error(`Fatal error: ${error.message}`)
  process.exit(1)
})
