# Facial Expression Analysis Pipeline

## Overview
This pipeline analyzes facial expressions in video recordings to calculate emotional diversity metrics for research in neurocognitive disorders. The complete workflow takes raw video files and produces publication-ready statistical analyses.

## Pipeline Components

### 1. modified_emotion_tracker.py
**Purpose:** Extract frame-by-frame emotion probabilities from videos  
**Input:** Video files (.avi format)  
**Output:** CSV files with emotion scores for each frame  

### 2. face_data_combine.py
**Purpose:** Aggregate frame-level data into participant-level metrics  
**Input:** CSV files from emotion_tracker.py  
**Output:** Excel/CSV with diversity indices and dispersion measures  

### 3. FaceDiversityAnalysis.py
**Purpose:** Statistical analysis comparing groups  
**Input:** Excel file from FaceCombine2.py + diagnosis information  
**Output:** Statistical tables, figures, and p-values  

## Quick Start

### Step 1: Process Videos
```bash
python emotion_tracker.py
```
- Place videos in nested folder structure
- Generates individual CSV files per video
- Creates master_analysis.csv with all data

### Step 2: Calculate Metrics
```bash
python FaceCombine2.py
```
- Reads all CSV files from Step 1
- Calculates Simpson's Diversity Index
- Computes dispersion measures
- Outputs emotion_analysis_summary.csv

### Step 3: Add Diagnosis Information
- Open emotion_analysis_summary.csv
- Add 'diagnosis' column with group labels (AD, bvFTD, MCI, HC)
- Save as Excel file

### Step 4: Run Statistical Analysis
```bash
python FaceDiversityAnalysis.py
```
- Reads Excel file with metrics and diagnoses
- Performs group comparisons
- Generates tables and figures
- Applies FDR correction

## Key Metrics Calculated

- **Emotional Diversity** - Simpson's Index measuring expression balance
- **Dispersion** - Variability in emotion confidence scores
- **Quartile Dispersion** - Robust measure of spread
- **Most Frequent Emotion** - Predominant expression

## Expected Outputs

1. **Table 3** - Group means, standard deviations, and p-values
2. **Table 4** - Correlations between diversity and dispersion
3. **Figure 1** - Violin plot of diversity by diagnosis
4. **Excel files** - Detailed statistical results

## Installation

```bash
# Create virtual environment
python -m venv facial_analysis_env
source facial_analysis_env/bin/activate  # On Windows: facial_analysis_env\Scripts\activate

# Install dependencies
pip install fer opencv-python tensorflow pandas numpy scipy statsmodels seaborn matplotlib tabulate openpyxl
```

## Directory Structure
```
project/
├── videos/
│   └── participant_folders/
├── emotion_results/
│   └── participant_csvs/
├── emotion_tracker.py
├── FaceCombine2.py
├── FaceDiversityAnalysis.py
└── output_files/
    ├── emotion_analysis_summary.csv
    ├── statistical_results.xlsx
    └── figures/
```

## Important Notes

- Only processes participants with role identifier "A" (typically patients)
- Requires face detection in frames (skips frames without faces)
- Statistical analysis requires at least 3 participants per group
- All p-values are FDR-corrected for multiple comparisons

## Troubleshooting

**No faces detected:**
- Check video quality and lighting
- Ensure faces are clearly visible
- Verify MTCNN is properly installed

**Missing data:**
- Script handles missing frames gracefully
- Participants with <100 valid frames may be excluded
- Check console output for skipped files

**Statistical errors:**
- Ensure all groups have sufficient participants
- Check for missing diagnosis labels
- Verify numeric data types in Excel

## Citation
If using this pipeline, please cite the original paper describing the diversity metrics approach to facial expression analysis in neurocognitive disorders.

## Attribution

This project is a heavily modified version of code originally developed by Susanta Biswas (© 2021) and licensed under the MIT License.

See `LICENSE.biswas` for the original license text.

## Dataset Attribution

This software builds upon models originally trained using the FER-2013 dataset, which is made available for non-commercial, academic research only.

FER-2013 dataset: [https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

## License

This project is released under the MIT License. See the [LICENSE](./LICENSE) file for details.

Note: Portions of the codebase are derived from prior work by Susanta Biswas, also under MIT. See `LICENSE.biswas` for original license terms.

This software incorporates models trained using the FER-2013 dataset, which is restricted to **non-commercial academic use only**. Redistribution or commercial use of trained models may be subject to additional limitations under the FER-2013 dataset license.

## Funding

This work was supported by the National Institutes of Health under grant number NIA K23 AG063900. The content is solely the responsibility of the authors and does not necessarily represent the official views of the NIH.
