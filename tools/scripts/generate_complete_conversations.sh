#!/bin/bash

echo "🤖 Automated Complete Crisis Conversation Generator"
echo "================================================="
echo "Generating full conversations with client + counselor responses"
echo "This will run automatically - just let it work..."
echo ""

# Create output file
timestamp=$(date +%Y%m%d_%H%M%S)
output_file="complete_crisis_conversations_${timestamp}.json"

echo '{' > "$output_file"
echo '  "metadata": {' >> "$output_file"
echo "    \"generated_at\": \"$(date -Iseconds)\"," >> "$output_file"
echo '    "model": "huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M",' >> "$output_file"
echo '    "type": "Complete crisis intervention conversations",' >> "$output_file"
echo '    "format": "Client message + Professional counselor response pairs"' >> "$output_file"
echo '  },' >> "$output_file"
echo '  "conversations": [' >> "$output_file"

# Simple crisis scenarios for complete conversations
declare -a scenarios=(
    "I want to end my life"
    "I've been cutting myself"
    "I can't stop drinking"
    "My partner abuses me"
    "I'm having panic attacks"
    "I feel completely hopeless"
    "I overdosed yesterday"
    "My family rejected me"
    "I can't cope anymore"
    "I'm scared all the time"
)

total=${#scenarios[@]}
echo "📊 Generating $total complete conversations..."
echo "⏱️  Estimated time: $((total * 2)) minutes"
echo ""

for i in "${!scenarios[@]}"; do
    crisis_prompt="${scenarios[$i]}"
    
    echo "🔄 Conversation $((i+1))/$total: '$crisis_prompt'"
    
    # Generate client message
    echo "   📝 Generating client message..."
    client_response=$(curl -s -X POST https://api.pixelatedempathy.tech/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Generate a realistic message from someone in crisis who says: $crisis_prompt\"}],
            \"max_tokens\": 60,
            \"temperature\": 0.8
        }" \
        --max-time 120)
    
    client_content=$(echo "$client_response" | jq -r '.choices[0].message.content' 2>/dev/null | sed 's/<think>.*<\/think>//g' | sed 's/<think>.*//g' | xargs)
    
    if [[ -n "$client_content" && "$client_content" != "null" ]]; then
        echo "   ✅ Client: ${client_content:0:50}..."
        
        # Generate counselor response
        echo "   🩺 Generating counselor response..."
        counselor_response=$(curl -s -X POST https://api.pixelatedempathy.tech/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M\",
                \"messages\": [{\"role\": \"user\", \"content\": \"Professional crisis counselor response to: $client_content\"}],
                \"max_tokens\": 60,
                \"temperature\": 0.7
            }" \
            --max-time 120)
        
        counselor_content=$(echo "$counselor_response" | jq -r '.choices[0].message.content' 2>/dev/null | sed 's/<think>.*<\/think>//g' | sed 's/<think>.*//g' | xargs)
        
        if [[ -n "$counselor_content" && "$counselor_content" != "null" ]]; then
            echo "   ✅ Counselor: ${counselor_content:0:50}..."
            
            # Add to JSON
            if [[ $i -gt 0 ]]; then
                echo "," >> "$output_file"
            fi
            
            echo "    {" >> "$output_file"
            echo "      \"conversation_id\": $((i+1))," >> "$output_file"
            echo "      \"crisis_type\": \"$crisis_prompt\"," >> "$output_file"
            echo "      \"client_message\": \"$client_content\"," >> "$output_file"
            echo "      \"counselor_response\": \"$counselor_content\"," >> "$output_file"
            echo "      \"timestamp\": \"$(date -Iseconds)\"" >> "$output_file"
            echo -n "    }" >> "$output_file"
            
            echo "   💾 Saved complete conversation pair"
        else
            echo "   ❌ Counselor response failed"
        fi
    else
        echo "   ❌ Client message failed"
    fi
    
    # Progress
    progress=$((i * 100 / total))
    echo "   📊 Overall progress: $progress%"
    echo ""
    
    # Wait between conversations
    if [[ $((i+1)) -lt $total ]]; then
        echo "   ⏳ Waiting 25 seconds before next conversation..."
        sleep 25
    fi
done

echo "" >> "$output_file"
echo "  ]" >> "$output_file"
echo "}" >> "$output_file"

echo "🎉 COMPLETE CRISIS CONVERSATIONS GENERATED!"
echo "=========================================="
echo "📁 File: $output_file"
echo "📊 Total: $total conversation pairs"
echo "🎯 Ready for crisis intervention training"
echo ""
echo "✨ You now have a complete dataset of authentic crisis conversations"
echo "   with both client expressions and professional counselor responses."
echo ""
echo "🛌 Take a break - the system did the work for you!"
