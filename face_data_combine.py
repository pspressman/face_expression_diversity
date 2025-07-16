import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.signal import find_peaks
import re

def calculate_emotional_metrics(emotion_matrix, timestamps):
    """
    Calculate comprehensive emotional variability metrics using multiple approaches.
    Each metric captures a different aspect of how emotions vary over time.
    
    Parameters:
        emotion_matrix: DataFrame containing emotion intensities over time
        timestamps: Series of timestamps for temporal analysis
    
    Returns:
        Dictionary of emotional variability metrics
    """
    metrics = {}
    
    # 1. Simpson's Diversity Index
    # This tells us how balanced someone's emotional expressions are
    # Higher values mean more balanced use of different emotions
    proportions = emotion_matrix.div(emotion_matrix.sum(axis=1), axis=0)
    simpson = (proportions ** 2).sum(axis=1)
    metrics['emotional_diversity'] = float((1 - simpson).mean())
    
    # 2. Dynamic Range Analysis
    # Captures the full spectrum of emotional expression
    # Both overall range and individual emotion ranges
    emotion_ranges = emotion_matrix.max() - emotion_matrix.min()
    metrics['overall_emotional_range'] = float(emotion_ranges.mean())
    for emotion in emotion_matrix.columns:
        metrics[f'{emotion}_range'] = float(emotion_ranges[emotion])
    
    # 3. Dispersion Metrics
    # Multiple approaches to measuring spread of emotional expression
    for emotion in emotion_matrix.columns:
        series = emotion_matrix[emotion]
        # Coefficient of Dispersion (robust to outliers)
        mad = np.abs(series - series.median()).median()
        metrics[f'{emotion}_dispersion'] = float(mad / (series.median() + 1e-6))
        
        # Quartile Coefficient of Dispersion
        q1, q3 = np.percentile(series, [25, 75])
        metrics[f'{emotion}_quartile_dispersion'] = float((q3 - q1) / (q3 + q1) if (q3 + q1) != 0 else 0)
    
    # 4. Temporal Dynamics
    # How emotions change over time
    emotion_changes = emotion_matrix.diff().abs()
    
    # Rate of change metrics
    metrics['mean_emotion_change'] = float(emotion_changes.mean().mean())
    metrics['max_emotion_change'] = float(emotion_changes.max().max())
    
    # 5. Emotional Stability Analysis
    # Looking at sustained emotional states
    threshold = 0.1  # Minimum change to consider significant
    sustained_periods = []
    for emotion in emotion_matrix.columns:
        changes = np.where(abs(np.diff(emotion_matrix[emotion])) > threshold)[0]
        if len(changes) > 0:
            periods = np.diff(changes)
            sustained_periods.extend(periods)
    
    metrics['avg_emotion_duration'] = float(np.mean(sustained_periods) if sustained_periods else 0)
    metrics['max_emotion_duration'] = float(np.max(sustained_periods) if sustained_periods else 0)
    
    # 6. Emotional Complexity Metrics
    # How many distinct emotional patterns appear
    for emotion in emotion_matrix.columns:
        # Find peaks in emotional intensity
        peaks, _ = find_peaks(emotion_matrix[emotion], height=0.2, distance=5)
        metrics[f'{emotion}_peak_count'] = len(peaks)
        if len(peaks) > 0:
            metrics[f'{emotion}_avg_peak_height'] = float(emotion_matrix[emotion].iloc[peaks].mean())
        else:
            metrics[f'{emotion}_avg_peak_height'] = 0.0
    
    # 7. Emotional Momentum
    # How much emotions carry forward (autocorrelation)
    for emotion in emotion_matrix.columns:
        # Calculate lag-1 autocorrelation
        autocorr = stats.pearsonr(emotion_matrix[emotion][:-1], 
                                emotion_matrix[emotion][1:])[0]
        metrics[f'{emotion}_autocorrelation'] = float(autocorr if not np.isnan(autocorr) else 0)
    
    # 8. Emotional Transition Analysis
    # Patterns of emotional changes
    dominant_emotions = emotion_matrix.idxmax(axis=1)
    transitions = pd.DataFrame({'from': dominant_emotions[:-1], 
                              'to': dominant_emotions[1:]})
    
    # Calculate transition matrix
    transition_matrix = pd.crosstab(transitions['from'], 
                                  transitions['to'], 
                                  normalize='index')
    
    # Store transition probabilities
    for from_emotion in transition_matrix.index:
        for to_emotion in transition_matrix.columns:
            prob = transition_matrix.loc[from_emotion, to_emotion]
            metrics[f'transition_prob_{from_emotion}_to_{to_emotion}'] = float(prob)
    
    return metrics

def extract_participant_id(filename):
    """
    Extract participant ID from filename, but only when participant is in role "A"
    
    Example matches:
    - PIDX-YYY-A (matches, returns "X-YYY")
    - PIDX-YYY-B (doesn't match)
    - X-YYY-A (matches, returns "X-YYY")
    - X-YYY-B (doesn't match)
    
    Returns the participant ID (e.g., 'X-YYY') or None if not found/error occurs
    """
    try:
        print(f"\nAnalyzing filename: {filename}")
        
        # Generic pattern to match participant IDs ending with role identifier
        # Adjust pattern based on your ID format (e.g., \d for digits, \w for alphanumeric)
        pattern = r'(?:PID)?(\w+-\w+)-A'
        match = re.search(pattern, filename)
        
        if not match:
            print(f"  Skipping: No participant ID pattern with role A found in filename")
            return None
            
        # Extract the matched portion
        participant_id = match.group(1)  # This gives us "X-YYY" portion
        print(f"  Found participant ID: {participant_id}")
        
        return participant_id
        
    except Exception as e:
        print(f"  Error processing filename {filename}: {e}")
        return None


def calculate_participant_summary(participant_data, emotion_cols):
    """
    Calculate key summary statistics for a single participant's emotional data.
    
    Parameters:
        participant_data: DataFrame containing all frames for one participant
        emotion_cols: List of emotion column names
    
    Returns:
        Dictionary of summary statistics focusing on emotional patterns
    """
    stats = {}
    
    # Basic recording statistics
    stats['total_duration'] = participant_data['timestamp'].max()
    stats['frame_count'] = len(participant_data)
    
    # For each emotion, calculate core statistics
    for emotion in emotion_cols:
        emotion_data = participant_data[emotion]
        
        # Central tendency and spread
        stats[f'{emotion}_mean'] = emotion_data.mean()
        stats[f'{emotion}_std'] = emotion_data.std()
        
        # Dynamic range
        stats[f'{emotion}_range'] = emotion_data.max() - emotion_data.min()
        
        # Rate of change (average absolute frame-to-frame difference)
        stats[f'{emotion}_volatility'] = emotion_data.diff().abs().mean()
    
    # Calculate most frequent dominant emotion
    emotion_matrix = participant_data[emotion_cols]
    dominant_emotions = emotion_matrix.idxmax(axis=1)
    stats['most_frequent_emotion'] = dominant_emotions.mode().iloc[0]
    stats['dominant_emotion_percentage'] = (dominant_emotions == stats['most_frequent_emotion']).mean() * 100
    
    return stats

def process_emotional_data(directory_path):
    """
    Process facial expression files to generate comprehensive summary statistics per participant.
    Uses a two-pass approach:
    1. First pass: Collect and temporally order all data for each participant
    2. Second pass: Calculate statistics on each participant's complete dataset
    """
    emotion_cols = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    participant_dataframes = {}  # Will store combined data for each participant
    
    # First pass: Collect and combine all data for each participant
    for csv_file in Path(directory_path).rglob('*.csv'):
        try:
            print(f"Reading file: {csv_file}")
            df = pd.read_csv(csv_file, on_bad_lines='skip')
            participant_id = extract_participant_id(csv_file.stem)
            
            if not participant_id or df.empty:
                continue
                
            df_valid = df[df['faces_found'] == True]
            
            # Add to participant's combined data with temporal ordering
            if participant_id in participant_dataframes:
                existing_data = participant_dataframes[participant_id].sort_values('timestamp')
                new_data = df_valid.sort_values('timestamp')
                participant_dataframes[participant_id] = pd.concat(
                    [existing_data, new_data]
                ).reset_index(drop=True)
            else:
                participant_dataframes[participant_id] = df_valid.sort_values(
                    'timestamp'
                ).reset_index(drop=True)
                
        except Exception as e:
            print(f"Error reading file {csv_file}: {e}")
            continue
    
    # Second pass: Calculate statistics on complete dataset for each participant
    participant_summaries = []
    
    for participant_id, full_participant_data in participant_dataframes.items():
        try:
            print(f"Calculating metrics for participant {participant_id}")
            
            # Calculate statistics on complete dataset
            basic_stats = calculate_participant_summary(full_participant_data, emotion_cols)
            emotion_matrix = full_participant_data[emotion_cols]
            advanced_stats = calculate_emotional_metrics(emotion_matrix, full_participant_data['timestamp'])
            
            stats = {
                'participant_id': participant_id,
                **basic_stats,
                **advanced_stats
            }
            
            participant_summaries.append(stats)
            print(f"Completed processing for participant {participant_id}")
            
        except Exception as e:
            print(f"Error calculating metrics for participant {participant_id}: {e}")
            continue
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(participant_summaries)
    
    return summary_df

def save_summary(summary_df, output_path, format='csv'):
    """
    Save the individual-level summary statistics with clear formatting.
    Creates separate sheets in Excel for different types of metrics.
    """
    # Round numerical values for clarity
    numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
    summary_df[numeric_cols] = summary_df[numeric_cols].round(4)
    
    if format.lower() == 'csv':
        summary_df.to_csv(output_path, index=False)
    else:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Write main summary sheet with all data
            summary_df.to_excel(writer, sheet_name='Individual Summaries', index=False)
            
            # Create separate sheets for different types of metrics
            # First, let's check what columns we actually have
            available_basic_cols = ['participant_id']
            if 'frame_count' in summary_df.columns:
                available_basic_cols.append('frame_count')
            if 'total_duration' in summary_df.columns:
                available_basic_cols.append('total_duration')
            
            # Write basic information sheet using only columns that exist
            summary_df[available_basic_cols].to_excel(writer, 
                                                    sheet_name='Basic Information', 
                                                    index=False)
            
            # Emotion summaries sheet
            emotion_cols = [col for col in summary_df.columns if any(
                emotion in col for emotion in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            )]
            if emotion_cols:  # Only create this sheet if we have emotion columns
                emotion_summary = summary_df[['participant_id'] + emotion_cols]
                emotion_summary.to_excel(writer, sheet_name='Emotion Summaries', index=False)
            
            # Variability metrics sheet
            variability_cols = [col for col in summary_df.columns if any(
                metric in col for metric in ['range', 'dispersion', 'duration', 'transition']
            )]
            if variability_cols:  # Only create this sheet if we have variability columns
                variability_summary = summary_df[['participant_id'] + variability_cols]
                variability_summary.to_excel(writer, sheet_name='Variability Metrics', index=False)

def main():
    # Replace with your directory containing the individual CSV files from emotion_tracker.py
    data_directory = "/path/to/your/emotion/results/"
    
    # Process all CSV files and calculate metrics
    summary_stats = process_emotional_data(data_directory)
    
    # Save outputs
    save_summary(summary_stats, "emotion_analysis_summary.csv")
    save_summary(summary_stats, "emotion_analysis_detailed.xlsx", format='xlsx')
    
    print("Comprehensive emotion analysis complete! Summary files have been created.")

if __name__ == "__main__":
    main()