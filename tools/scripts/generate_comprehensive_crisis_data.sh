#!/bin/bash

echo "🚀 Comprehensive Crisis Training Dataset Generation"
echo "=================================================="

# Create output file
timestamp=$(date +%Y%m%d_%H%M%S)
output_file="comprehensive_crisis_dataset_${timestamp}.json"

echo '{' > "$output_file"
echo '  "metadata": {' >> "$output_file"
echo "    \"generated_at\": \"$(date -Iseconds)\"," >> "$output_file"
echo '    "model": "huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M",' >> "$output_file"
echo '    "purpose": "Crisis intervention training data",' >> "$output_file"
echo '    "note": "Generated using abliterated model for authentic crisis language"' >> "$output_file"
echo '  },' >> "$output_file"
echo '  "crisis_scenarios": [' >> "$output_file"

# Comprehensive crisis prompts covering different scenarios
declare -a prompts=(
    # Suicidal ideation
    "I want to kill myself"
    "I have a plan to end my life"
    "I don't want to be alive anymore"
    "Nobody would miss me if I died"
    
    # Self-harm
    "I've been cutting myself"
    "I hurt myself when I'm upset"
    "I can't stop harming myself"
    
    # Depression
    "I feel completely empty inside"
    "Nothing matters anymore"
    "I can't get out of bed"
    "I'm worthless"
    
    # Anxiety/Panic
    "I can't breathe, I'm panicking"
    "I'm having a panic attack"
    "I'm terrified and don't know why"
    
    # Domestic violence
    "My partner hits me"
    "I'm afraid to go home"
    "He said he'll kill me if I leave"
    
    # Substance abuse
    "I can't stop drinking"
    "I overdosed last night"
    "I need drugs to function"
    
    # LGBTQ+ crisis
    "My family disowned me for being gay"
    "I hate who I am"
    "I'm confused about my identity"
    
    # General crisis
    "I need help right now"
    "Everything is falling apart"
    "I don't know what to do"
)

declare -a categories=(
    "suicidal_ideation"
    "suicidal_ideation" 
    "suicidal_ideation"
    "suicidal_ideation"
    "self_harm"
    "self_harm"
    "self_harm"
    "depression"
    "depression"
    "depression"
    "depression"
    "anxiety_panic"
    "anxiety_panic"
    "anxiety_panic"
    "domestic_violence"
    "domestic_violence"
    "domestic_violence"
    "substance_abuse"
    "substance_abuse"
    "substance_abuse"
    "lgbtq_crisis"
    "lgbtq_crisis"
    "lgbtq_crisis"
    "general_crisis"
    "general_crisis"
    "general_crisis"
)

total=${#prompts[@]}
echo "📊 Generating $total crisis scenarios..."

for i in "${!prompts[@]}"; do
    prompt="${prompts[$i]}"
    category="${categories[$i]}"
    
    echo ""
    echo "📝 Processing $((i+1))/$total: '$prompt'"
    echo "   Category: $category"
    
    # Make API call
    response=$(curl -s -X POST https://api.pixelatedempathy.tech/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}],
            \"max_tokens\": 50,
            \"temperature\": 0.8
        }" \
        --max-time 120)
    
    if [[ $? -eq 0 ]] && [[ -n "$response" ]]; then
        # Extract content from response
        content=$(echo "$response" | jq -r '.choices[0].message.content' 2>/dev/null)
        
        if [[ "$content" != "null" ]] && [[ -n "$content" ]]; then
            # Clean up thinking tags
            cleaned_content=$(echo "$content" | sed 's/<think>.*<\/think>//g' | sed 's/<think>.*//g' | xargs)
            
            echo "   ✅ Generated: ${cleaned_content:0:60}..."
            
            # Add to JSON (with comma if not first)
            if [[ $i -gt 0 ]]; then
                echo "," >> "$output_file"
            fi
            
            echo "    {" >> "$output_file"
            echo "      \"id\": $((i+1))," >> "$output_file"
            echo "      \"category\": \"$category\"," >> "$output_file"
            echo "      \"crisis_prompt\": \"$prompt\"," >> "$output_file"
            echo "      \"model_response\": \"$content\"," >> "$output_file"
            echo "      \"cleaned_response\": \"$cleaned_content\"," >> "$output_file"
            echo "      \"timestamp\": \"$(date -Iseconds)\"," >> "$output_file"
            echo "      \"intensity_level\": 8" >> "$output_file"
            echo -n "    }" >> "$output_file"
        else
            echo "   ❌ Failed to extract content"
        fi
    else
        echo "   ❌ API call failed"
    fi
    
    # Progress indicator
    progress=$((i * 100 / total))
    echo "   📊 Progress: $progress%"
    
    # Wait between requests to avoid overwhelming server
    if [[ $((i+1)) -lt $total ]]; then
        echo "   ⏳ Waiting 20 seconds..."
        sleep 20
    fi
done

echo "" >> "$output_file"
echo "  ]" >> "$output_file"
echo "}" >> "$output_file"

echo ""
echo "🎉 COMPREHENSIVE CRISIS DATASET GENERATION COMPLETE!"
echo "=================================================="
echo "📁 Dataset saved to: $output_file"
echo "📊 Total scenarios processed: $total"
echo "🔍 Check the file for detailed crisis training data"
echo ""
echo "This dataset contains authentic crisis language generated by an"
echo "abliterated model for training crisis intervention systems."
