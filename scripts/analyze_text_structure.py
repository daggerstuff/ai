#!/usr/bin/env python3
"""
Analyze the actual text structure around the question
"""

def analyze_text():
    text = "And I guess to take this one step further, I've heard you talk about that one of the main problems with society today is that a lot of people who have mental health struggles and they're struggling with situations this where they feel stuck and they can't get out of their own way at times. They feel people are against them. They're dealing with trauma. And then they might go see somebody who's not trained in trauma. They don't have the experience. How can somebody begin to take that path and make sure that they're finding somebody that is trained in that and then also that they're able to self-regulate themselves. When needed. , that's a huge question because unfortunately, look, I've been through medical training."
    
    print("=== TEXT STRUCTURE ANALYSIS ===\n")
    
    # Find the "How can" part
    how_can_pos = text.find("How can")
    if how_can_pos != -1:
        print(f"'How can' found at position: {how_can_pos}")
        
        # Show context around it
        context_start = max(0, how_can_pos - 50)
        context_end = min(len(text), how_can_pos + 200)
        context = text[context_start:context_end]
        
        print(f"Context around 'How can':")
        print(f"'{context}'")
        print()
        
        # Look for question mark
        question_mark_pos = text.find("?", how_can_pos)
        if question_mark_pos != -1:
            print(f"Question mark found at position: {question_mark_pos}")
            full_question = text[how_can_pos:question_mark_pos + 1]
            print(f"Full question: '{full_question}'")
        else:
            print("No question mark found after 'How can'")
            
            # Look for sentence endings
            sentence_endings = [". ", "! ", "? "]
            for ending in sentence_endings:
                ending_pos = text.find(ending, how_can_pos)
                if ending_pos != -1:
                    print(f"Sentence ending '{ending}' found at position: {ending_pos}")
                    potential_question = text[how_can_pos:ending_pos + 1]
                    print(f"Potential question: '{potential_question}'")
                    break

if __name__ == "__main__":
    analyze_text()
