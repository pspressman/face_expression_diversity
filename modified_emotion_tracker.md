# emotion_tracker.py

Facial Expression Analysis Pipeline
Overview
Python implementation for batch processing of facial expression analysis across multiple videos stored in a nested directory structure. Uses the FER (Facial Expression Recognition) library with MTCNN face detection.
Dependencies
pythonCopyfrom fer import FER
import pandas as pd
import cv2
from pathlib import Path
import gc
import os
Main Class: BatchTracker
Initialization
pythonCopydef __init__(self, output_dir, batch_size=100):
    self.face_detector = FER(mtcnn=True)
    self.output_dir = Path(output_dir)
    self.output_dir.mkdir(exist_ok=True)
    self.batch_size = batch_size
Key Methods
Directory Scanning
pythonCopydef find_video_files(self, base_dir):
    """
    Recursively finds .avi/.AVI files in directory structure.
    Searches participant folders (2-XXX/Video Files/).
    Returns: List of (participant_id, video_path) tuples
    """
Batch Processing
pythonCopydef process_batch(self, batch_frames, batch_indices, fps, participant_id, video_name):
    """
    Processes a batch of frames for face detection and emotion analysis.
    Returns: List of dictionaries containing frame analysis
    """
Video Processing
pythonCopydef process_video(self, video_path, participant_id):
    """
    Processes single video in batches.
    - Creates participant-specific output directory
    - Saves individual video CSV
    - Updates master CSV
    """
Directory Processing
pythonCopydef process_directory(self, base_dir):
    """
    Main processing pipeline.
    - Scans directory structure
    - Processes each video
    - Handles errors and continues processing
    """
Usage
pythonCopytracker = BatchTracker(
    output_dir="/Users/username/MyDevelopment/emotion_results",
    batch_size=100
)
    
tracker.process_directory("/path/to/video/directory")
Output Structure

emotion_results/

master_analysis.csv
participant_id/

video_name_analysis.csv





Data Format
Output CSV Columns

participant_id
video
frame
timestamp
face_number
total_faces
x, y (face position)
faces_found
Emotion scores:

angry
disgust
fear
happy
sad
surprise
neutral



Memory Management

Processes videos in 100-frame batches
Clears memory after each batch
Forces garbage collection
Saves progress incrementally

Error Handling

Continues processing if individual video fails
Maintains data for successfully processed videos
Reports errors without stopping pipeline

Directory Structure Requirements
Expected structure:
Copybase_directory/
    2-XXX/
        Video Files/
            video1.avi
            video2.AVI
            Follow-Up/
                video3.avi
Running the Code
bashCopy# Create and activate virtual environment
python3 -m venv emotion_venv
source emotion_venv/bin/activate

# Install dependencies
pip install fer pandas opencv-python tensorflow numpy mtcnn keras

# Run analysis
python emotion_tracker.py
