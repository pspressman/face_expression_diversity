from fer import FER
import pandas as pd
import cv2
from pathlib import Path
import gc
import os

class BatchTracker:
    def __init__(self, output_dir, batch_size=100):
        self.face_detector = FER(mtcnn=True)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.batch_size = batch_size
        
    def find_video_files(self, base_dir):
        """
        Carefully find all video files in the nested structure.
        Returns list of (participant_id, video_path) tuples.
        """
        video_files = []
        base_path = Path(base_dir)
        
        # Print what we're scanning
        print(f"\nScanning for videos in: {base_path}")
        
        try:
            # Look through numbered participant folders
            for participant_folder in sorted(base_path.glob("[2-9]-[0-9][0-9][0-9]")):
                participant_id = participant_folder.name
                video_folder = participant_folder / "Video Files"
                
                if video_folder.exists() and video_folder.is_dir():
                    print(f"\nChecking {participant_id} - Video Files folder found")
                    
                    # Find all .avi files (case insensitive)
                    for video_file in video_folder.glob("*.[aA][vV][iI]"):
                        print(f"Found: {video_file.name}")
                        video_files.append((participant_id, video_file))
                        
                    # Also check Follow-Up folder if it exists
                    followup = video_folder / "Follow-Up"
                    if followup.exists() and followup.is_dir():
                        for video_file in followup.glob("*.[aA][vV][iI]"):
                            print(f"Found in Follow-Up: {video_file.name}")
                            video_files.append((participant_id, video_file))
                
        except Exception as e:
            print(f"Error scanning directories: {e}")
            raise
            
        print(f"\nTotal videos found: {len(video_files)}")
        return video_files

    def process_batch(self, batch_frames, batch_indices, fps, participant_id, video_name):
        """Process a batch of frames together."""
        batch_data = []
        
        for frame, frame_idx in zip(batch_frames, batch_indices):
            timestamp = frame_idx / fps
            faces = self.face_detector.detect_emotions(frame)
            
            if faces:
                for face_idx, face in enumerate(faces):
                    batch_data.append({
                        'participant_id': participant_id,
                        'video': video_name,
                        'frame': frame_idx,
                        'timestamp': timestamp,
                        'face_number': face_idx + 1,
                        'total_faces': len(faces),
                        'x': face['box'][0],
                        'y': face['box'][1],
                        'faces_found': True,
                        **face['emotions']
                    })
            else:
                batch_data.append({
                    'participant_id': participant_id,
                    'video': video_name,
                    'frame': frame_idx,
                    'timestamp': timestamp,
                    'face_number': 0,
                    'total_faces': 0,
                    'x': None, 'y': None,
                    'faces_found': False,
                    'angry': None, 'disgust': None,
                    'fear': None, 'happy': None,
                    'sad': None, 'surprise': None,
                    'neutral': None
                })
        
        return batch_data
    
    def process_video(self, video_path, participant_id):
        """Process video in fixed-size batches."""
        video_name = video_path.name
        print(f"\nProcessing {participant_id} - {video_name}")
        
        # Create participant directory for results if needed
        participant_dir = self.output_dir / participant_id
        participant_dir.mkdir(exist_ok=True)
        output_file = participant_dir / f"{video_name}_analysis.csv"
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        first_batch = True
        
        try:
            while frame_count < total_frames:
                # Collect batch
                batch_frames = []
                batch_indices = []
                
                for _ in range(self.batch_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    batch_frames.append(rgb_frame)
                    batch_indices.append(frame_count)
                    frame_count += 1
                
                if not batch_frames:
                    break
                
                # Process batch
                batch_data = self.process_batch(batch_frames, batch_indices, fps, 
                                             participant_id, video_name)
                df = pd.DataFrame(batch_data)
                
                # Save to individual video CSV
                if first_batch:
                    df.to_csv(output_file, index=False)
                    first_batch = False
                else:
                    df.to_csv(output_file, mode='a', header=False, index=False)
                
                # Save to master CSV
                master_file = self.output_dir / 'master_analysis.csv'
                if not master_file.exists():
                    df.to_csv(master_file, index=False)
                else:
                    df.to_csv(master_file, mode='a', header=False, index=False)
                
                # Clear batch data
                del batch_frames, batch_indices, batch_data, df
                gc.collect()
                
                print(f"Processed {frame_count}/{total_frames} frames", end='\r')
                
        finally:
            cap.release()
            
        print(f"\nCompleted {participant_id} - {video_name}")
        gc.collect()
    
    def process_directory(self, base_dir):
        """Process all videos in nested directory structure."""
        print(f"Starting analysis of videos in {base_dir}")
        
        try:
            # First, find all videos
            video_files = self.find_video_files(base_dir)
            
            if not video_files:
                print("No video files found!")
                return
                
            # Process each video
            for idx, (participant_id, video_path) in enumerate(video_files, 1):
                print(f"\nProcessing video {idx}/{len(video_files)}")
                print(f"Participant: {participant_id}")
                print(f"Video: {video_path.name}")
                
                try:
                    self.process_video(video_path, participant_id)
                except Exception as e:
                    print(f"Error processing {video_path.name}: {e}")
                    # Continue with next video
                    continue
                
            print("\nProcessing complete!")
            print(f"Results saved in: {self.output_dir}")
            
        except Exception as e:
            print(f"Error in directory processing: {e}")
            raise

# Example usage
if __name__ == "__main__":
    tracker = BatchTracker(
        output_dir="/Users/peterpressman/MyDevelopment/emotion_results",
        batch_size=100
    )
    
    tracker.process_directory("/Volumes/easystore/conversational speech 18-0456/Participant Data and Forms")
