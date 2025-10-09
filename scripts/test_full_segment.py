#!/usr/bin/env python3
"""
Test smart agent with full segment text displayed
"""

import sys
sys.path.append('/root/pixelated/ai/scripts')
from smart_qa_agent import SmartQAAgent

def test_with_full_text():
    """Test with complete segment text"""
    agent = SmartQAAgent()
    
    # Full segment from the data
    test_segment = {
        "text": "And I guess to take this one step further, I've heard you talk about that one of the main problems with society today is that a lot of people who have mental health struggles and they're struggling with situations this where they feel stuck and they can't get out of their own way at times. They feel people are against them. They're dealing with trauma. And then they might go see somebody who's not trained in trauma. They don't have the experience. How can somebody begin to take that path and make sure that they're finding somebody that is trained in that and then also that they're able to self-regulate themselves. When needed. , that's a huge question because unfortunately, look, I've been through medical training. I'm a physician. And the average physician never hears any of the stuff I just talked about. The average psychiatrist doesn't get any training in trauma. Not in. They learn something about PTSD, which is a specific form of trauma. But they don't learn about the traumatic basis of depression and anxiety and ADHD. And they learn nothing about it. it's very difficult to find good help within the medical system. Now, many therapists also don't get any such training. There's a lot of therapists that are designed only to change your beliefs and your behaviors, but not to address the fundamental reasons for those behaviors. a lot of psychologists trained in CBT, cognitive behavioral therapy, or dialectical behavioral therapy. A lot of them are not really. And I know this. Believe me, I know this. They just don't know much about or anything about trauma. Then they can't help you with the fundamental wound that you're carrying. They can help you with the manifestations. And that's not useless. But they can't help you heal at your core. then there are therapies that are deeper than that. There is body-based therapies such as somatic experiencing developed by my friend and teacher, Dr. Peter Levine. There's sensory motor psychotherapy developed by Pact Ogden. There's EMDR that works for some people. There's internal family systems. There's a lot of different therapies that are developed by my friend and colleague, Dr. Richard Schwartz. There's compassionate inquiry, which is based on my work. And I train therapists in that method. There are others, other names I could mention.",
        "style": "therapeutic",
        "confidence": 3.076923076923077,
        "quality": 0.7,
        "source": "doug_bopst",
        "file": "give_me_15_minutes_i_ll_save_you_25plus_years_of_feeling_lonely_depressed_and_lost_dr_gabor_mat.txt"
    }
    
    result = agent.process_segment(test_segment)
    
    print("=== FULL SEGMENT ANALYSIS ===\n")
    print(f"**Content Type**: {result['smart_analysis']['content_type']}")
    print(f"**Dialogue Structure**: {result['smart_analysis']['dialogue_structure']}")
    print(f"**Main Topic**: {result['smart_analysis']['main_topic']}")
    print(f"**Confidence**: {result['smart_analysis']['confidence']}")
    
    if result['smart_analysis'].get('question_embedded'):
        print(f"**Embedded Question Found**: {result['smart_analysis']['question_embedded']}")
    
    print(f"\n**Generated Question**: {result['input']}")
    print(f"\n**Full Answer**:")
    print(result['output'])

if __name__ == "__main__":
    test_with_full_text()
