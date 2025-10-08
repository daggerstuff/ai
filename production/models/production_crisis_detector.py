#!/usr/bin/env python3
"""
Enhanced V5 Crisis Detector - Production Wrapper
Production-ready interface for crisis detection system
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from crisis_detector_v5_production import EnhancedCrisisDetectorV5
import logging
from datetime import datetime

class ProductionCrisisDetector:
    """Production wrapper for Enhanced V5"""
    
    def __init__(self):
        self.detector = EnhancedCrisisDetectorV5()
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup production logging"""
        logger = logging.getLogger('crisis_detector_production')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('../logs/crisis_detection.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def detect_crisis(self, text: str, user_id: str = None) -> dict:
        """Production crisis detection with logging"""
        try:
            result = self.detector.detect_crisis(text)
            
            # Log detection
            self.logger.info(f"Crisis detection - User: {user_id}, Crisis: {result['is_crisis']}, Confidence: {result['confidence']:.3f}")
            
            # Add production metadata
            result['production_timestamp'] = datetime.now().isoformat()
            result['production_version'] = 'V5_Production'
            result['user_id'] = user_id
            
            return result
            
        except Exception as e:
            self.logger.error(f"Crisis detection error - User: {user_id}, Error: {str(e)}")
            return {
                'is_crisis': False,
                'confidence': 0.0,
                'error': str(e),
                'production_timestamp': datetime.now().isoformat(),
                'production_version': 'V5_Production_Error'
            }

# Production instance
production_detector = ProductionCrisisDetector()

def detect_crisis(text: str, user_id: str = None) -> dict:
    """Main production interface"""
    return production_detector.detect_crisis(text, user_id)

if __name__ == "__main__":
    # Test production system
    test_cases = [
        "I want to kill myself tonight",
        "I'm feeling sad but I'll be okay",
        "Life is meaningless and hopeless"
    ]
    
    print("Enhanced V5 Production System Test")
    print("=" * 50)
    
    for i, text in enumerate(test_cases, 1):
        result = detect_crisis(text, f"test_user_{i}")
        print(f"Test {i}: {text[:30]}...")
        print(f"  Crisis: {result['is_crisis']}")
        print(f"  Confidence: {result.get('confidence', 0):.3f}")
        print()
