# FaceDiversityAnalysis.py

## Statistical Analysis Pipeline for Facial Expression Diversity

### Overview
Python implementation for comprehensive statistical analysis of facial expression diversity metrics across diagnostic groups. Analyzes emotional diversity, dispersion measures, and predominant emotions in neurocognitive disorders (AD, bvFTD, MCI, HC).

### Dependencies
```python
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from tabulate import tabulate
from scipy.stats import pearsonr, kruskal, mannwhitneyu
```

### Input Data Requirements

#### Excel File Structure
- **One row per participant**
- **Required columns:**
  - `diagnosis` - Group membership: 'FTD', 'AD', 'MCI', 'HC'
  - `emotional_diversity` - Simpson's Diversity Index
  - `most_frequent_emotion` - Predominant emotion category
  - `{emotion}_dispersion` - Dispersion for each emotion
  - `{emotion}_quartile_dispersion` - Quartile dispersion for each emotion

#### Emotion Categories
- angry, disgust, fear, happy, sad, surprise, neutral

### Key Functions

#### Data Loading and Validation
```python
def load_and_validate_data(file_path):
    """
    Loads Excel file and validates required columns.
    Returns: dataframe, numerical_vars, categorical_vars
    """
```

#### Primary Outcome Analysis
```python
def create_primary_outcome_table(df):
    """
    Creates Table 3 from the paper.
    - Calculates means ± SD for each group
    - Performs one-way ANOVA
    - Applies FDR correction
    Returns: formatted table, comparison results
    """
```

#### Correlation Analysis
```python
def correlation_with_diversity_analysis(df):
    """
    Generates Table 4 - correlations with emotional diversity.
    - Pearson correlations for all dispersion measures
    - FDR correction for multiple comparisons
    Returns: correlation table with p-values
    """
```

#### Categorical Emotion Analysis
```python
def analyze_most_frequent_emotion(df):
    """
    Chi-square analysis of predominant emotions.
    - Overall group differences
    - Post-hoc Fisher's exact tests
    - Percentage distributions
    Returns: contingency table, percentages, statistics
    """
```

#### Group Comparisons
```python
def calculate_numerical_differences(df, variables):
    """
    Pairwise group comparisons for continuous variables.
    - Kruskal-Wallis for overall differences
    - Mann-Whitney U for pairwise comparisons
    - Cohen's d effect sizes
    - FDR correction
    """
```

### Visualization Functions

#### Primary Outcome Visualization
```python
def plot_emotional_diversity_violin(df):
    """
    Creates Figure 1 - violin plot with embedded boxplot.
    Shows distribution of emotional diversity by diagnosis.
    """
```

#### Dispersion Visualizations
```python
def plot_quartile_dispersions(df):
    """
    Creates boxplots and violin plots for:
    - Happy quartile dispersion
    - Sad quartile dispersion
    """
```

### Statistical Methods

1. **Group Differences**
   - One-way ANOVA (F-tests) for overall differences
   - Mann-Whitney U tests for pairwise comparisons
   - Kruskal-Wallis for non-parametric analysis

2. **Effect Sizes**
   - Cohen's d for continuous variables
   - Cramer's V for categorical variables

3. **Multiple Comparison Correction**
   - Benjamini-Hochberg FDR procedure
   - Applied to all p-values

4. **Correlation Analysis**
   - Pearson correlations
   - Focus on dispersion measures
   - Identifies potential mediators

### Output Files

- `primary_outcome_table_with_fdr.xlsx` - Main results table
- `diversity_correlations.xlsx` - Correlation analysis
- `emotion_analysis_results.xlsx` - Categorical emotion analysis
- `group_difference_results.xlsx` - Detailed group comparisons
- `potential_mediators.xlsx` - Variables correlated with diversity
- Multiple `.png` files for visualizations

### Usage

```python
# Set file path
file_path = '/path/to/your/data.xlsx'

# Run main analysis
if __name__ == "__main__":
    main()
```

### Analysis Pipeline

1. **Load and validate data**
   - Checks for required columns
   - Identifies variable types
   - Reports missing data

2. **Primary outcomes**
   - Emotional diversity analysis
   - Dispersion measures for all emotions
   - Group comparisons with FDR correction

3. **Correlation analysis**
   - Identifies dispersion measures correlated with diversity
   - Tests for group differences in correlated variables

4. **Categorical analysis**
   - Predominant emotion distributions
   - Chi-square and post-hoc tests

5. **Visualization**
   - Generates all figures for publication
   - Creates diagnostic plots

### Key Results Generated

- **Table 3**: Primary outcomes with means ± SD and p-values
- **Table 4**: Correlations between diversity and dispersion measures
- **Figure 1**: Violin plot of emotional diversity by group
- **Supplementary figures**: Dispersion distributions

### Running the Analysis

```bash
# Install required packages
pip install pandas numpy scipy statsmodels seaborn matplotlib tabulate openpyxl

# Run analysis
python FaceDiversityAnalysis.py
```

### Notes

- Handles missing data appropriately
- Includes extensive error checking
- Produces publication-ready tables and figures
- All statistical tests match those described in the paper

### Data Flow Summary

```
1. emotion_tracker.py
   ├── Input: Video files (.avi)
   └── Output: Frame-by-frame emotion probabilities (CSV)
       └── Columns: frame, timestamp, angry, disgust, fear, happy, sad, surprise, neutral

2. [Intermediate aggregation step - not shown]
   ├── Input: Frame-by-frame CSVs
   ├── Calculations:
   │   ├── Simpson's Diversity Index
   │   ├── Dispersion (MAD/median)
   │   ├── Quartile dispersion ((Q3-Q1)/(Q3+Q1))
   │   └── Most frequent emotion
   └── Output: Participant-level metrics (Excel)

3. FaceDiversityAnalysis.py
   ├── Input: Aggregated Excel file
   └── Output: Statistical results, tables, figures
```
