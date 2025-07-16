# FaceCombine2.py

## Participant-Level Aggregation Pipeline for Facial Expression Data

### Overview
Python implementation for aggregating frame-by-frame emotion data into participant-level metrics. Calculates Simpson's Diversity Index, dispersion measures, and other statistical summaries needed for downstream analysis of facial expression patterns in neurocognitive disorders.

### Dependencies
```python
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.signal import find_peaks
import re
```

### Input Requirements

#### Directory Structure
```
emotion_results/
├── participant_id/
│   ├── video1_analysis.csv
│   ├── video2_analysis.csv
│   └── ...
└── ...
```

#### CSV File Format (from emotion_tracker.py)
Each CSV must contain:
- `participant_id` - Participant identifier
- `timestamp` - Time of frame
- `faces_found` - Boolean indicating if face was detected
- `angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral` - Emotion probabilities (0-1)

### Code Assumptions

1. **Participant ID Format**
   - IDs follow pattern: `X-YYY-R` where R is role identifier
   - Only processes participants with role "A" (e.g., patients)
   - Skips participants with other roles (e.g., partners)

2. **Data Quality**
   - Only analyzes frames where `faces_found == True`
   - Handles missing or corrupted files gracefully
   - Combines multiple videos per participant chronologically

3. **Temporal Ordering**
   - Sorts data by timestamp before analysis
   - Handles multiple recording sessions per participant

### Key Functions

#### Metric Calculation
```python
def calculate_emotional_metrics(emotion_matrix, timestamps):
    """
    Calculates comprehensive emotional variability metrics.
    
    Metrics include:
    1. Simpson's Diversity Index
    2. Dynamic range for each emotion
    3. Dispersion coefficients
    4. Temporal dynamics
    5. Emotional stability measures
    6. Peak analysis
    7. Autocorrelation
    8. Transition probabilities
    
    Returns: Dictionary of all metrics
    """
```

#### Participant ID Extraction
```python
def extract_participant_id(filename):
    """
    Extracts participant ID from filename.
    Pattern: PIDX-YYY-A or X-YYY-A
    Only matches role "A" participants.
    
    Returns: Participant ID or None
    """
```

#### Summary Statistics
```python
def calculate_participant_summary(participant_data, emotion_cols):
    """
    Basic statistics per participant:
    - Duration and frame count
    - Mean, std, range for each emotion
    - Volatility (frame-to-frame changes)
    - Most frequent emotion
    """
```

### Calculated Metrics

#### Primary Metrics (for paper)
- `emotional_diversity` - Simpson's Diversity Index (1 - Σp²)
- `{emotion}_dispersion` - Median absolute deviation / median
- `{emotion}_quartile_dispersion` - (Q3-Q1)/(Q3+Q1)
- `most_frequent_emotion` - Mode of dominant emotions

#### Additional Metrics
- `{emotion}_range` - Max - Min for each emotion
- `{emotion}_volatility` - Average absolute change
- `{emotion}_peak_count` - Number of intensity peaks
- `{emotion}_autocorrelation` - Lag-1 correlation
- `transition_prob_{from}_to_{to}` - Emotion transition matrix

### Processing Pipeline

1. **First Pass - Data Collection**
   ```python
   # Reads all CSV files
   # Extracts participant IDs
   # Combines data per participant
   # Maintains temporal order
   ```

2. **Second Pass - Metric Calculation**
   ```python
   # For each participant:
   # - Calculate basic statistics
   # - Compute diversity metrics
   # - Generate transition matrices
   # - Store all results
   ```

3. **Output Generation**
   - CSV file with all metrics
   - Excel file with organized sheets

### Output Files

#### emotion_analysis_summary.csv
Single CSV with one row per participant containing all metrics.

#### emotion_analysis_detailed.xlsx
Excel file with multiple sheets:
- `Individual Summaries` - All data
- `Basic Information` - ID, duration, frame count
- `Emotion Summaries` - Emotion-specific metrics
- `Variability Metrics` - Dispersion and range measures

### Usage

```python
def main():
    # Set input directory
    data_directory = "/path/to/your/emotion/results/"
    
    # Process all participants
    summary_stats = process_emotional_data(data_directory)
    
    # Save outputs
    save_summary(summary_stats, "emotion_analysis_summary.csv")
    save_summary(summary_stats, "emotion_analysis_detailed.xlsx", format='xlsx')

if __name__ == "__main__":
    main()
```

### Running the Script

```bash
# Install dependencies
pip install pandas numpy scipy openpyxl

# Run aggregation
python FaceCombine2.py
```

### Data Flow Integration

```
1. emotion_tracker.py
   └── Frame-by-frame emotion probabilities (CSV)
       ↓
2. FaceCombine2.py (this script)
   └── Participant-level metrics (CSV/Excel)
       ↓
3. FaceDiversityAnalysis.py
   └── Statistical analysis and results
```

### Key Calculations

#### Simpson's Diversity Index
```python
proportions = emotion_matrix.div(emotion_matrix.sum(axis=1), axis=0)
simpson = (proportions ** 2).sum(axis=1)
diversity = (1 - simpson).mean()
```

#### Dispersion Coefficient
```python
mad = np.abs(series - series.median()).median()
dispersion = mad / (series.median() + 1e-6)
```

#### Quartile Dispersion
```python
q1, q3 = np.percentile(series, [25, 75])
quartile_dispersion = (q3 - q1) / (q3 + q1)
```

### Notes

- Handles participants with multiple recording sessions
- Robust to missing data and file errors
- Produces metrics directly comparable to published literature
- All calculations match those described in the paper's methods section