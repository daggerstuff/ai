#!/usr/bin/env python3
"""
Proper Dialogue Parser and Q/A Extractor
Correctly parses interview/dialogue structure and creates coherent Q/A pairs
"""

import json
import re
from typing import Dict, List, Tuple, Optional

class DialogueParser:
    def __init__(self):
        pass
    
    def identify_speakers(self, text: str) -> List[Dict]:
        """Identify different speakers and their segments in the text"""
        
        # Look for explicit speaker markers
        speaker_patterns = [
            r'(HOST|INTERVIEWER|GUEST|TIM|DR\.|THERAPIST):\s*(.+?)(?=(?:HOST|INTERVIEWER|GUEST|TIM|DR\.|THERAPIST):|$)',
            r'(Q|QUESTION):\s*(.+?)(?=(?:A|ANSWER):|$)',
            r'(A|ANSWER):\s*(.+?)(?=(?:Q|QUESTION):|$)'
        ]
        
        segments = []
        
        for pattern in speaker_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                speaker = match.group(1).strip()
                content = match.group(2).strip()
                segments.append({
                    "speaker": speaker,
                    "content": content,
                    "start": match.start(),
                    "end": match.end()
                })
        
        # If no explicit markers, try to detect dialogue flow
        if not segments:
            segments = self.detect_implicit_dialogue(text)
        
        return sorted(segments, key=lambda x: x.get('start', 0))
    
    def detect_implicit_dialogue(self, text: str) -> List[Dict]:
        """Detect dialogue structure without explicit markers"""
        
        # Look for question-answer patterns
        sentences = re.split(r'[.!?]+', text)
        segments = []
        
        current_speaker = "unknown"
        current_content = []
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Detect questions
            if '?' in sentence or any(word in sentence.lower() for word in ['how', 'what', 'why', 'when', 'can you', 'would you']):
                # This might be a question
                if current_content and current_speaker != "interviewer":
                    # Save previous content
                    segments.append({
                        "speaker": current_speaker,
                        "content": '. '.join(current_content),
                        "start": 0,
                        "end": 0
                    })
                
                current_speaker = "interviewer"
                current_content = [sentence]
            else:
                # This is likely a response
                if current_speaker == "interviewer":
                    # Save question
                    segments.append({
                        "speaker": "interviewer", 
                        "content": '. '.join(current_content),
                        "start": 0,
                        "end": 0
                    })
                    current_speaker = "tim_fletcher"
                    current_content = [sentence]
                else:
                    current_content.append(sentence)
        
        # Add final segment
        if current_content:
            segments.append({
                "speaker": current_speaker,
                "content": '. '.join(current_content),
                "start": 0,
                "end": 0
            })
        
        return segments
    
    def extract_qa_pairs(self, segments: List[Dict]) -> List[Tuple[str, str]]:
        """Extract coherent Q/A pairs from dialogue segments"""
        
        qa_pairs = []
        
        for i in range(len(segments) - 1):
            current = segments[i]
            next_segment = segments[i + 1]
            
            # Look for question followed by answer
            if (self.is_question(current['content']) and 
                self.is_answer(next_segment['content']) and
                current['speaker'] != next_segment['speaker']):
                
                question = current['content'].strip()
                answer = next_segment['content'].strip()
                
                # Validate the pair makes sense
                if self.validate_qa_pair(question, answer):
                    qa_pairs.append((question, answer))
        
        return qa_pairs
    
    def is_question(self, text: str) -> bool:
        """Determine if text is a question"""
        text_lower = text.lower().strip()
        
        # Explicit question markers
        if '?' in text:
            return True
            
        # Question words
        question_starters = ['how', 'what', 'why', 'when', 'where', 'who', 'can you', 'would you', 'could you', 'do you']
        if any(text_lower.startswith(starter) for starter in question_starters):
            return True
            
        return False
    
    def is_answer(self, text: str) -> bool:
        """Determine if text is an answer/response"""
        text_lower = text.lower().strip()
        
        # Answer indicators
        answer_starters = ['well', 'so', 'yes', 'no', 'absolutely', 'that\'s', 'the thing is', 'what happens']
        
        # Should be substantial content
        if len(text.split()) < 10:
            return False
            
        return True
    
    def validate_qa_pair(self, question: str, answer: str) -> bool:
        """Validate that question and answer are coherent"""
        
        # Basic length checks
        if len(question.split()) < 3 or len(answer.split()) < 10:
            return False
            
        # Check for topic coherence (basic keyword overlap)
        q_words = set(re.findall(r'\b\w+\b', question.lower()))
        a_words = set(re.findall(r'\b\w+\b', answer.lower()))
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'you', 'i', 'we', 'they', 'he', 'she', 'it'}
        
        q_keywords = q_words - common_words
        a_keywords = a_words - common_words
        
        # Check for some thematic overlap
        overlap = len(q_keywords & a_keywords)
        if overlap > 0 or len(a_keywords) > 20:  # Either overlap or substantial answer
            return True
            
        return False
    
    def process_segment(self, segment: Dict) -> Optional[Dict]:
        """Process a segment to extract proper Q/A pair"""
        
        text = segment['text']
        
        # Parse dialogue structure
        dialogue_segments = self.identify_speakers(text)
        
        # Extract Q/A pairs
        qa_pairs = self.extract_qa_pairs(dialogue_segments)
        
        if qa_pairs:
            # Use the first valid Q/A pair
            question, answer = qa_pairs[0]
            
            return {
                "input": question,
                "output": answer,
                "style": segment['style'],
                "confidence": segment['confidence'],
                "quality": segment['quality'],
                "source": segment['source'],
                "file": segment['file'],
                "parsing_info": {
                    "method": "dialogue_extraction",
                    "segments_found": len(dialogue_segments),
                    "qa_pairs_found": len(qa_pairs)
                }
            }
        
        # If no dialogue structure found, create contextual question
        return self.create_contextual_fallback(segment)
    
    def create_contextual_fallback(self, segment: Dict) -> Dict:
        """Create contextual question when dialogue parsing fails"""
        
        text = segment['text']
        style = segment['style']
        
        # Analyze what the text is actually explaining
        if "narcissist" in text.lower() and "manipulat" in text.lower():
            question = "Why do narcissists become manipulative?"
        elif "trauma" in text.lower() and "therap" in text.lower():
            question = "What should someone know about trauma therapy?"
        elif "shame" in text.lower():
            question = "Can you explain how shame affects people?"
        elif "heal" in text.lower():
            question = "What does the healing process look like?"
        else:
            # Generic contextual question
            if style == "therapeutic":
                question = "Can you share your therapeutic perspective on this?"
            elif style == "educational":
                question = "Can you explain this concept?"
            elif style == "empathetic":
                question = "Can you help me understand this?"
            else:
                question = "What should I know about this?"
        
        return {
            "input": question,
            "output": text,
            "style": segment['style'],
            "confidence": segment['confidence'],
            "quality": segment['quality'],
            "source": segment['source'],
            "file": segment['file'],
            "parsing_info": {
                "method": "contextual_fallback",
                "reason": "no_dialogue_structure_detected"
            }
        }

def test_dialogue_parser():
    """Test the dialogue parser with the problematic segment"""
    parser = DialogueParser()
    
    test_segment = {
        "text": "And I guess to take this one step further, I've heard you talk about that one of the main problems with society today is that a lot of people who have mental health struggles and they're struggling with situations this where they feel stuck and they can't get out of their own way at times. They feel people are against them. They're dealing with trauma. And then they might go see somebody who's not trained in trauma. They don't have the experience. How can somebody begin to take that path and make sure that they're finding somebody that is trained in that and then also that they're able to self-regulate themselves. When needed. , that's a huge question because unfortunately, look, I've been through medical training. I'm a physician. And the average physician never hears any of the stuff I just talked about. The average psychiatrist doesn't get any training in trauma.",
        "style": "therapeutic",
        "confidence": 3.0,
        "quality": 0.7,
        "source": "interview",
        "file": "test.txt"
    }
    
    result = parser.process_segment(test_segment)
    
    print("=== DIALOGUE PARSER TEST ===\n")
    print(f"**Method**: {result['parsing_info']['method']}")
    print(f"**Q**: {result['input']}")
    print(f"**A**: {result['output'][:300]}...")
    print(f"**Parsing Info**: {result['parsing_info']}")

if __name__ == "__main__":
    test_dialogue_parser()
