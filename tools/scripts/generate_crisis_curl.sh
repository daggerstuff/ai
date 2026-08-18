#!/bin/bash

echo "🚀 Generating Crisis Training Data with Direct API Calls"
echo "========================================================"

# Create output file
timestamp=$(date +%Y%m%d_%H%M%S)
output_file="crisis_data_${timestamp}.json"

echo '{"crisis_responses": [' > "$output_file"

# Crisis prompts
prompts=("I feel hopeless" "I'm in pain" "Help me" "I'm scared" "I can't cope")

for i in "${!prompts[@]}"; do
    prompt="${prompts[$i]}"
    echo ""
    echo "📝 Generating response $((i+1))/5 for: '$prompt'"
    
    # Make API call
    response=$(curl -s -X POST https://api.pixelatedempathy.com/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}],
            \"max_tokens\": 30
        }" \
        --max-time 120)
    
    if [[ $? -eq 0 ]] && [[ -n "$response" ]]; then
        # Extract content from response
        content=$(echo "$response" | jq -r '.choices[0].message.content' 2>/dev/null)
        
        if [[ "$content" != "null" ]] && [[ -n "$content" ]]; then
            echo "✅ Success: $content"
            
            # Add to JSON (with comma if not first)
            if [[ $i -gt 0 ]]; then
                echo "," >> "$output_file"
            fi
            
            echo "  {" >> "$output_file"
            echo "    \"id\": $((i+1))," >> "$output_file"
            echo "    \"prompt\": \"$prompt\"," >> "$output_file"
            echo "    \"response\": \"$content\"," >> "$output_file"
            echo "    \"timestamp\": \"$(date -Iseconds)\"" >> "$output_file"
            echo -n "  }" >> "$output_file"
        else
            echo "❌ Failed to extract content"
        fi
    else
        echo "❌ API call failed"
    fi
    
    echo "⏳ Waiting 15 seconds..."
    sleep 15
done

echo "" >> "$output_file"
echo "]}" >> "$output_file"

echo ""
echo "🎉 Generation complete!"
echo "📁 Data saved to: $output_file"
echo "📊 Check the file for results"

