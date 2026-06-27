import json
import subprocess
from pathlib import Path
import sys

# We will stream the master dataset from S3 and write locally
# Then we will push the results to S3

def run():
    print("Streaming master dataset...")
    # For now, let's just see how many records we can process in 10 seconds
    pass

if __name__ == '__main__':
    run()
