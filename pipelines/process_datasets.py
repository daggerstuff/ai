#!/usr/bin/env python3
"""
Pixelated Empathy AI - Data Processing Pipeline
Production-ready data processing for empathy datasets
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipelines.data_processing.empathy_processor import EmpathyProcessor
from pipelines.data_processing.validation import DataValidator
from config.production.data_config import DataConfig

def setup_logging():
    """Configure logging for data processing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipelines/logs/processing.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main data processing pipeline"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Pixelated Empathy Data Processing Pipeline")
    
    try:
        # Load configuration
        config = DataConfig.from_file('config/production/data.yaml')
        
        # Initialize processor
        processor = EmpathyProcessor(config)
        
        # Initialize validator
        validator = DataValidator(config.validation_config)
        
        # Process datasets
        processed_data = processor.process_all()
        
        # Validate processed data
        validation_results = validator.validate(processed_data)
        
        if validation_results.is_valid:
            logger.info("Data processing completed successfully")
        else:
            logger.error(f"Data validation failed: {validation_results.errors}")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Data processing failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
