import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from tabulate import tabulate
from scipy.stats import pearsonr, kruskal, mannwhitneyu

def preprocess_data(df):
    """
    Preprocess the dataframe to handle missing values and data types
    """
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Replace various forms of missing data with numpy NaN
    missing_values = ['', '.', 'NA', 'na', 'N/A', 'n/a', None]
    df_clean.replace(missing_values, np.nan, inplace=True)
    
    return df_clean

def analyze_most_frequent_emotion(df):
    """
    Analyze most_frequent_emotion using chi-square test
    """
    print("\nAnalyzing most frequent emotions across groups:")
    
    # Create contingency table
    contingency = pd.crosstab(df['diagnosis'], df['most_frequent_emotion'])
    print("\nEmotion frequencies by group:")
    print(contingency)
    
    # Perform chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    print("\nPerforming post-hoc analysis...")
    posthoc_results = perform_emotion_posthoc(contingency)
    
    print(f"\nChi-square test results:")
    print(f"Chi-square statistic: {chi2:.2f}")
    print(f"p-value: {p_value:.3f}")
    print(f"Degrees of freedom: {dof}")
    
    # Calculate percentages within each group
    percentages = contingency.div(contingency.sum(axis=1), axis=0) * 100
    print("\nPercentages within each group:")
    print(percentages)
    
    return {
        'contingency': contingency,
        'percentages': percentages,
        'chi2': chi2,
        'p_value': p_value,
        'posthoc': posthoc_results
    }

def plot_variable_distributions(df, variables, group_col='diagnosis'):
    """
    Plot boxplots and histograms for numerical variables across diagnostic groups
    """
    for var in variables:
        if var not in df.columns:
            print(f"Skipping {var} - not in dataframe")
            continue
        
        plt.figure(figsize=(12, 5))

        # Boxplot
        plt.subplot(1, 2, 1)
        sns.boxplot(x=group_col, y=var, data=df, palette='Set3')
        plt.title(f"Boxplot of {var} by {group_col}")
        plt.xticks(rotation=45)

        # Histogram
        plt.subplot(1, 2, 2)
        sns.histplot(data=df, x=var, hue=group_col, element='step', stat='density', common_norm=False)
        plt.title(f"Histogram of {var} by {group_col}")
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(f"{var}_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

def load_and_validate_data(file_path):
    """
    Load data from Excel file and validate contents
    """
    print("STARTING load_and_validate_data")
    try:
        df = pd.read_excel(file_path)
        
        expected_groups = {'FTD', 'AD', 'MCI', 'HC'}
        
        if 'diagnosis' not in df.columns:
            raise ValueError("Missing 'diagnosis' column in the dataset")
        
        found_groups = set(df['diagnosis'].unique())
        if not expected_groups.issubset(found_groups):
            missing = expected_groups - found_groups
            raise ValueError(f"Missing diagnostic groups: {missing}")
            
        emotion_vars = [col for col in df.columns if any(emotion in col for emotion in 
                      ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'emotion'])]
        
        if not emotion_vars:
            raise ValueError("No emotion-related variables found in the dataset")
        # Add after finding emotion_vars but before categorizing them:
        
        print("\nDiagnostic information:")
            
        for var in emotion_vars:
            print(f"\n{var}:")
            print(f"Data type: {df[var].dtype}")
            try:
                # Get unique values without sorting first
                unique_vals = df[var].unique()
                print(f"First few unique values (unsorted): {unique_vals[:5]}")
                print(f"Total number of unique values: {len(unique_vals)}")
            except Exception as e:
                print(f"Error getting unique values: {str(e)}")
        
        # Variable type detection
        categorical_vars = []
        numerical_vars = []
        for var in emotion_vars:
            # Try to convert to numeric
            try:
                pd.to_numeric(df[var].dropna())
                numerical_vars.append(var)
            except:
                # If conversion fails, it's likely categorical
                unique_vals = df[var].dropna().unique()
                # Print info about what we found
                print(f"\nVariable {var} has unique values: {unique_vals}")
                categorical_vars.append(var)
        
        print(f"\nNumerical variables: {numerical_vars}")
        print(f"Categorical variables: {categorical_vars}")
        
        # Check for missing values
        missing_data = df[emotion_vars].isnull().sum()
        if missing_data.any():
            print("\nWarning: Missing values found in the following variables:")
            print(missing_data[missing_data > 0])
            
        return df, numerical_vars, categorical_vars
        
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def calculate_numerical_differences(df, variables, group_col='diagnosis', groups=None):
    """
    Calculate group differences for numerical variables
    """
    if groups is None:
        groups = sorted(df[group_col].unique())
    
    group_pairs = list(combinations(groups, 2))
    results = pd.DataFrame(index=variables, columns=[f"{g1}_vs_{g2}" for g1, g2 in group_pairs])
    effect_sizes = results.copy()
    
    for var in variables:
        groups_data = [df[df[group_col] == g][var].dropna().values for g in groups]
        
        if any(len(gd) == 0 for gd in groups_data):
            print(f"Warning: Skipping {var} due to insufficient data in one or more groups")
            for pair in [f"{g1}_vs_{g2}" for g1, g2 in group_pairs]:
                results.loc[var, pair] = np.nan
                effect_sizes.loc[var, pair] = np.nan
            continue
            
        try:
            # Check if all values in any group are identical
            if any(len(np.unique(gd)) == 1 for gd in groups_data):
                print(f"Warning: Skipping {var} - some groups have identical values")
                for pair in [f"{g1}_vs_{g2}" for g1, g2 in group_pairs]:
                    results.loc[var, pair] = 1  # Set p-value to 1 for no significance
                    effect_sizes.loc[var, pair] = 0  # Set effect size to 0
                continue
            
            try:
                h_stat, h_pval = stats.kruskal(*groups_data)
            except ValueError as e:
                print(f"Warning: Skipping {var} - {str(e)}")
                for pair in [f"{g1}_vs_{g2}" for g1, g2 in group_pairs]:
                    results.loc[var, pair] = 1
                    effect_sizes.loc[var, pair] = 0
                continue
                
            if h_pval < 0.05:
                for idx, (g1, g2) in enumerate(group_pairs):
                    group1_data = df[df[group_col] == g1][var].dropna()
                    group2_data = df[df[group_col] == g2][var].dropna()
                    
                    stat, pval = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                    results.loc[var, f"{g1}_vs_{g2}"] = pval
                    
                    d = (np.mean(group1_data) - np.mean(group2_data)) / np.sqrt(
                        ((len(group1_data) - 1) * np.var(group1_data) + 
                         (len(group2_data) - 1) * np.var(group2_data)) / 
                        (len(group1_data) + len(group2_data) - 2))
                    effect_sizes.loc[var, f"{g1}_vs_{g2}"] = d
            else:
                for pair in [f"{g1}_vs_{g2}" for g1, g2 in group_pairs]:
                    results.loc[var, pair] = 1
                    effect_sizes.loc[var, pair] = 0
        except Exception as e:
            print(f"Warning: Error processing {var}: {str(e)}")
            for pair in [f"{g1}_vs_{g2}" for g1, g2 in group_pairs]:
                results.loc[var, pair] = np.nan
                effect_sizes.loc[var, pair] = np.nan
    
    # Convert results to numpy array and handle non-numeric values
    flat_pvals = pd.to_numeric(results.values.flatten(), errors='coerce')
    mask = ~np.isnan(flat_pvals)
    
    if mask.any():
        fwe_corrected = np.full_like(flat_pvals, np.nan)
        valid_pvals = flat_pvals[mask]
        if len(valid_pvals) > 0:
            fwe_corrected[mask] = multipletests(flat_pvals[mask], method='fdr_bh')[0]
        fwe_results = pd.DataFrame(fwe_corrected.reshape(results.shape),
                                 index=results.index,
                                 columns=results.columns)
    else:
        fwe_results = results.copy()
    
    return {
        'unadjusted': results,
        'fwe_adjusted': fwe_results,
        'effect_sizes': effect_sizes
    }

def calculate_categorical_differences(df, variables, group_col='diagnosis', groups=None):
    """
    Calculate group differences for categorical variables using chi-square tests
    """
    if groups is None:
        groups = sorted(df[group_col].unique())
    
    group_pairs = list(combinations(groups, 2))
    results = pd.DataFrame(index=variables, columns=[f"{g1}_vs_{g2}" for g1, g2 in group_pairs])
    effect_sizes = results.copy()
    
    for var in variables:
        for g1, g2 in group_pairs:
            try:
                # Create contingency table
                contingency = pd.crosstab(
                    df[df[group_col].isin([g1, g2])][group_col],
                    df[df[group_col].isin([g1, g2])][var]
                )
                
                # Chi-square test
                chi2, pval = stats.chi2_contingency(contingency)[:2]
                results.loc[var, f"{g1}_vs_{g2}"] = pval
                
                # Cramer's V as effect size
                n = contingency.sum().sum()
                min_dim = min(contingency.shape) - 1
                cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else np.nan
                effect_sizes.loc[var, f"{g1}_vs_{g2}"] = cramers_v
                
            except Exception as e:
                print(f"Warning: Error processing {var} for {g1} vs {g2}: {str(e)}")
                results.loc[var, f"{g1}_vs_{g2}"] = np.nan
                effect_sizes.loc[var, f"{g1}_vs_{g2}"] = np.nan
    
    # FWE correction
    flat_pvals = pd.to_numeric(results.values.flatten(), errors='coerce')
    mask = ~np.isnan(flat_pvals)
    
    if mask.any():
        fwe_corrected = np.full_like(flat_pvals, np.nan)
        valid_pvals = flat_pvals[mask]
        if len(valid_pvals) > 0:
            fwe_corrected[mask] = multipletests(flat_pvals[mask], method='fdr_bh')[0]
        fwe_results = pd.DataFrame(fwe_corrected.reshape(results.shape),
                                 index=results.index,
                                 columns=results.columns)
    else:
        fwe_results = results.copy()
    

def plot_heatmap(p_values, effect_sizes, title, figsize=(15, 20)):
    # Convert effect sizes to numeric, replacing non-numeric values with NaN
    effect_sizes_numeric = pd.DataFrame(effect_sizes).apply(pd.to_numeric, errors='coerce')
    
    sig_markers = p_values.copy()
    sig_markers[p_values.isna()] = 'NA'
    sig_markers[p_values > 0.05] = ''
    sig_markers[p_values <= 0.05] = '*'
    sig_markers[p_values <= 0.01] = '**'
    sig_markers[p_values <= 0.001] = '***'
    
    plt.figure(figsize=figsize)
    sns.heatmap(effect_sizes_numeric,  # Use the numeric version
                annot=sig_markers,
                fmt='',
                cmap='RdBu_r',
                center=0,
                vmin=-1.5,
                vmax=1.5,
                cbar_kws={'label': "Effect Size"})

    
    plt.title(title)
    plt.ylabel('Variables')
    plt.xlabel('Group Comparisons')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return plt.gcf()

def perform_emotion_posthoc(contingency):
    """
    Perform post-hoc analysis for emotion frequencies using Fisher's exact test
    """
    groups = contingency.index
    emotions = contingency.columns
    results = []
    
    # Perform Fisher's exact test for each pair of groups
    for g1, g2 in combinations(groups, 2):
        for emotion in emotions:
            # Create 2x2 contingency table for this emotion vs all others
            emotion_table = np.zeros((2, 2))
            # This emotion
            emotion_table[0,0] = contingency.loc[g1, emotion]
            emotion_table[1,0] = contingency.loc[g2, emotion]
            # All other emotions
            emotion_table[0,1] = contingency.loc[g1].sum() - emotion_table[0,0]
            emotion_table[1,1] = contingency.loc[g2].sum() - emotion_table[1,0]
            
            # Perform Fisher's exact test
            _, p_value = stats.fisher_exact(emotion_table)
            
            results.append({
                'Group 1': g1,
                'Group 2': g2,
                'Emotion': emotion,
                'p_value': p_value
            })
    
    # Convert to DataFrame and adjust for multiple comparisons
    results_df = pd.DataFrame(results)
    results_df['p_adj'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
    
    # Print significant results
    print("\nPost-hoc analysis results (FDR-corrected):")
    sig_results = results_df[results_df['p_adj'] < 0.05].sort_values('p_adj')
    if len(sig_results) > 0:
        for _, row in sig_results.iterrows():
            print(f"\n{row['Group 1']} vs {row['Group 2']} - {row['Emotion']}:")
            print(f"p-value (adjusted): {row['p_adj']:.3f}")
    else:
        print("No significant post-hoc comparisons found after correction")
    
    return results_df
    
    # Convert to DataFrame and adjust for multiple comparisons
    results_df = pd.DataFrame(results)
    results_df['p_adj'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
    
    # Print significant results
    print("\nPost-hoc analysis results (FDR-corrected):")
    sig_results = results_df[results_df['p_adj'] < 0.05].sort_values('p_adj')
    if len(sig_results) > 0:
        for _, row in sig_results.iterrows():
            print(f"\n{row['Group 1']} vs {row['Group 2']} - {row['Emotion']}:")
            print(f"p-value (adjusted): {row['p_adj']:.3f}")
            print(f"Effect size (Phi): {row['Effect_Size']:.3f}")
    else:
        print("No significant post-hoc comparisons found after correction")
    
    return results_df

def plot_emotional_diversity(df, var='emotional_diversity', group_col='diagnosis'):
    """
    Create a boxplot showing emotional diversity across diagnostic groups
    """
    if var not in df.columns:
        print(f"Variable '{var}' not found in dataframe.")
        return
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=group_col, y=var, data=df, palette='Set2')
    sns.stripplot(x=group_col, y=var, data=df, color='black', alpha=0.4, jitter=0.2)
    plt.title('Emotional Diversity Across Diagnostic Groups')
    plt.ylabel('Emotional Diversity Index')
    plt.xlabel('Diagnosis')
    plt.tight_layout()
    plt.savefig('emotional_diversity_by_diagnosis.png', dpi=300)
    plt.close()

def plot_emotional_diversity_violin(df, var='emotional_diversity', group_col='diagnosis'):
    """
    Create a violin + boxplot for emotional diversity across diagnostic groups
    """
    if var not in df.columns:
        print(f"Variable '{var}' not found in dataframe.")
        return
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=group_col, y=var, data=df, inner=None, palette='Set2', cut=0)
    sns.boxplot(x=group_col, y=var, data=df, width=0.2, color='black', showfliers=False)
    sns.stripplot(x=group_col, y=var, data=df, color='black', alpha=0.3, jitter=0.2)

    plt.title('Emotional Diversity Across Diagnostic Groups (Violin + Boxplot)')
    plt.ylabel('Emotional Diversity Index')
    plt.xlabel('Diagnosis')
    plt.tight_layout()
    plt.savefig('emotional_diversity_violin.png', dpi=300)
    plt.close()

from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

def correlation_with_pvalues(df, target='emotional_diversity', candidates=None):
    """
    Compute Pearson correlation and p-value between the target variable
    and all other numeric variables in the dataframe
    """
    if candidates is None:
        candidates = [col for col in df.select_dtypes(include='number').columns if col != target]

    results = []
    for var in candidates:
        try:
            valid_data = df[[target, var]].dropna()
            if len(valid_data) > 2:
                r, p = pearsonr(valid_data[target], valid_data[var])
                results.append({'Variable': var, 'Correlation': r, 'p-value': p})
        except Exception as e:
            print(f"Error with variable {var}: {str(e)}")

    result_df = pd.DataFrame(results)

    # Apply FDR correction to the p-values
    if not result_df.empty:
        valid_pvals = result_df['p-value'].dropna().astype(float)
        adjusted = multipletests(valid_pvals, method='fdr_bh')[1]
        result_df.loc[valid_pvals.index, 'FDR_Adjusted'] = adjusted

    return result_df.sort_values('Correlation', key=abs, ascending=False)

def correlation_and_group_diff_summary(df, target='emotional_diversity', group_col='diagnosis'):
    """
    Generate a summary table of top correlates of emotional diversity
    with their correlation, p-value, and group-level difference stats
    """
    numeric_vars = [col for col in df.select_dtypes(include=np.number).columns if col != target]
    
    summary = []
    
    for var in numeric_vars:
        print(f"\nChecking {var}...")

        # Step 1: Correlation
        valid_corr = df[[target, var]].dropna()
        if len(valid_corr) < 3:
            print(f"  ➤ Skipping: too few valid observations for correlation.")
            continue
        try:
            r, p_corr = pearsonr(valid_corr[target], valid_corr[var])
        except Exception as e:
            print(f"  ➤ Skipping: correlation failed due to {e}")
            continue

        # Step 2: Kruskal-Wallis setup
        groups = df[group_col].dropna().unique()
        group_data = [df[df[group_col] == g][var].dropna() for g in groups]
        if any(len(g) < 3 for g in group_data):
            print(f"  ➤ Skipping: too few values in at least one group.")
            continue
        if any(np.all(g == g.iloc[0]) for g in group_data if len(g) > 0):
            print(f"  ➤ Skipping: constant values in at least one group.")
            continue

        # Step 3: Kruskal-Wallis test
        try:
            h_stat, p_kruskal = kruskal(*group_data)
        except Exception as e:
            print(f"  ➤ Skipping: Kruskal-Wallis failed due to {e}")
            continue

        # Step 4: Post-hoc Mann-Whitney + Cohen’s d
        group_pairs = list(combinations(groups, 2))
        pvals = []
        ds = []
        for g1, g2 in group_pairs:
            data1 = df[df[group_col] == g1][var].dropna()
            data2 = df[df[group_col] == g2][var].dropna()
            if len(data1) < 3 or len(data2) < 3:
                continue
            try:
                stat, pval = mannwhitneyu(data1, data2, alternative='two-sided')
                pvals.append(pval)
                d = (np.mean(data1) - np.mean(data2)) / np.sqrt(
                    ((len(data1) - 1) * np.var(data1, ddof=1) +
                     (len(data2) - 1) * np.var(data2, ddof=1)) /
                    (len(data1) + len(data2) - 2)
                )
                ds.append(abs(d))
            except Exception as e:
                print(f"    ➤ Skipping pair {g1} vs {g2}: {e}")
                continue

        if not pvals:
            print("  ➤ Skipping: no valid pairwise tests.")
            continue

        adj_pvals = multipletests(pvals, method='fdr_bh')[1]
        
        summary.append({
            'Variable': var,
            'Correlation': r,
            'Corr_p': p_corr,
            'Kruskal_p': p_kruskal,
            'Any_Sig_Group_Diff': np.any(adj_pvals < 0.05),
            'Smallest_p': np.min(adj_pvals),
            'Max_Cohens_d': np.max(ds) if ds else np.nan
        })

    result_df = pd.DataFrame(summary)
        
    if not result_df.empty:
        return result_df.sort_values(by='Correlation', key=np.abs, ascending=False)
    else:
        print("No valid results to summarize.")
        return result_df

def correlation_with_diversity_analysis(df, target='emotional_diversity'):
    """
    Perform analysis of correlations between emotional diversity
    and dispersion measures for all emotions.
    """
    # Focus ONLY on dispersion patterns
    emotion_patterns = ['_dispersion', '_quartile_dispersion']
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    # Generate list of ONLY dispersion variables to analyze
    candidates = []
    for emotion in emotions:
        for pattern in emotion_patterns:
            var = f"{emotion}{pattern}"
            if var in df.columns:
                candidates.append(var)
    
    print(f"Analyzing correlations between {target} and {len(candidates)} dispersion measures")
    
    # Calculate correlations and p-values
    results = []
    valid_pvals = []
    valid_indices = []
    
    for i, var in enumerate(candidates):
        try:
            valid_data = df[[target, var]].dropna()
            if len(valid_data) > 2:
                r, p = pearsonr(valid_data[target], valid_data[var])
                results.append({'Variable': var, 'Correlation': r, 'p-value': p})
                valid_pvals.append(p)
                valid_indices.append(i)
        except Exception as e:
            print(f"Error with variable {var}: {str(e)}")

    # Apply FDR correction to valid p-values
    if valid_pvals:
        reject, adjusted_pvals, _, _ = multipletests(valid_pvals, method='fdr_bh')
        
        # Add adjusted p-values to results
        for i, idx in enumerate(valid_indices):
            results[idx]['FDR_Adjusted'] = adjusted_pvals[i]
            results[idx]['Significant'] = reject[i]
    
    # Convert to DataFrame and sort by absolute correlation value
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('Correlation', key=abs, ascending=False)
    
    # Save results to Excel
    result_df.to_excel('diversity_correlations.xlsx', index=False)
    
    # Print top correlations
    print("\nTop correlations with emotional diversity:")
    top_results = result_df.head(10)
    print(tabulate(top_results, headers="keys", tablefmt="grid"))
    
    return result_df

def identify_potential_mediators(correlation_df, group_diff_df):
    """
    Identify variables that both:
    1. Significantly correlate with emotional diversity
    2. Show significant group differences
    """
    # Filter correlation results to significant ones
    sig_correlations = correlation_df[correlation_df['Significant'] == True]
    
    # Merge with group differences
    mediator_candidates = []
    
    for _, corr_row in sig_correlations.iterrows():
        var = corr_row['Variable']
        if var in group_diff_df['Variable'].values:
            diff_row = group_diff_df[group_diff_df['Variable'] == var].iloc[0]
            
            mediator_candidates.append({
                'Variable': var,
                'Correlation': corr_row['Correlation'],
                'Corr_FDR_p': corr_row['FDR_Adjusted'],
                'Group_Diff_FDR_p': diff_row['FDR p-value'] if 'FDR p-value' in diff_row else None,
                'FDR_Significant': diff_row['FDR Significant'] if 'FDR Significant' in diff_row else None
            })
    
    mediator_df = pd.DataFrame(mediator_candidates)
    
    if not mediator_df.empty:
        print("\nPotential mediators (variables with both significant correlation and group differences):")
        print(tabulate(mediator_df, headers="keys", tablefmt="grid"))
        mediator_df.to_excel('potential_mediators.xlsx', index=False)
    else:
        print("\nNo variables found that both correlate with diversity and show group differences.")
    
    return mediator_df

def create_primary_outcome_table(df):
    """
    Create a comprehensive table for primary outcome variables including
    emotional diversity and dispersion metrics for all emotions,
    with FDR-corrected p-values.
    """
    from statsmodels.stats.multitest import multipletests

    # Include all emotions in the analysis
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    # Generate primary outcome variables: diversity + dispersion for all emotions
    primary_vars = ['emotional_diversity']
    for emotion in emotions:
        primary_vars.append(f'{emotion}_dispersion')
        primary_vars.append(f'{emotion}_quartile_dispersion')

    # Descriptive stats
    group_stats = df.groupby('diagnosis')[primary_vars].agg(['mean', 'std', 'count'])

    raw_p_values = []
    comparison_results = {}

    for var in primary_vars:
        if var not in df.columns:
            print(f"Warning: {var} not found in dataset, skipping")
            continue
            
        groups = [df[df['diagnosis'] == group][var].dropna() for group in df['diagnosis'].unique()]
        try:
            # Check if all values in any group are identical
            if any(len(np.unique(gd)) == 1 for gd in groups if len(gd) > 0):
                print(f"Warning: Skipping {var} - some groups have identical values")
                comparison_results[var] = {'F_statistic': float('nan'), 'p_value': float('nan')}
                raw_p_values.append(float('nan'))
                continue
                
            f_stat, p_value = stats.f_oneway(*groups)
        except:
            f_stat, p_value = float('nan'), float('nan')
            
        comparison_results[var] = {'F_statistic': f_stat, 'p_value': p_value}
        raw_p_values.append(p_value)

    # Remove NaN values before FDR correction
    valid_indices = [i for i, p in enumerate(raw_p_values) if not np.isnan(p)]
    valid_p_values = [raw_p_values[i] for i in valid_indices]
    
    # Apply FDR correction to valid p-values
    if valid_p_values:
        reject, corrected_pvals, _, _ = multipletests(valid_p_values, alpha=0.05, method='fdr_bh')
        
        # Put corrected values back in original array
        all_corrected_pvals = [float('nan')] * len(raw_p_values)
        all_reject = [False] * len(raw_p_values)
        for i, idx in enumerate(valid_indices):
            all_corrected_pvals[idx] = corrected_pvals[i]
            all_reject[idx] = reject[i]
    else:
        all_corrected_pvals = raw_p_values
        all_reject = [False] * len(raw_p_values)

    # Final table
    table_data = []
    valid_vars = []
    
    for i, var in enumerate(primary_vars):
        if var not in df.columns:
            continue
            
        row = {'Variable': var}
        for group in df['diagnosis'].unique():
            group_data = df[df['diagnosis'] == group][var].dropna()
            if len(group_data) > 0:
                m = group_data.mean()
                s = group_data.std()
                n = len(group_data)
                row[group] = f"{m:.2f} ± {s:.2f} (n={int(n)})"
            else:
                row[group] = "N/A"

        raw_p = raw_p_values[i]
        if np.isnan(raw_p):
            continue
            
        corr_p = all_corrected_pvals[i]

        row['Raw p-value'] = "p < 0.001" if raw_p < 0.001 else f"p = {raw_p:.3f}"
        row['FDR p-value'] = "p < 0.001" if corr_p < 0.001 else f"p = {corr_p:.3f}"
        row['FDR Significant'] = "Yes" if all_reject[i] else "No"

        table_data.append(row)
        valid_vars.append(var)

    outcome_table = pd.DataFrame(table_data)
    outcome_table.to_excel('primary_outcome_table_with_fdr.xlsx', index=False)

    return outcome_table, comparison_results

"""
def create_primary_outcome_table(df):

    #Create a comprehensive table for primary outcome variables including
    #emotional diversity, total dispersion, and quartile dispersion metrics.
  
    # Primary outcome variables
    primary_vars = [
        'emotional_diversity',
        'happy_dispersion',
        'happy_quartile_dispersion',
        'sad_dispersion',
        'sad_quartile_dispersion'
    ]
    
    # Calculate descriptive statistics by group
    group_stats = df.groupby('diagnosis')[primary_vars].agg(['mean', 'std', 'count'])
    
    # Perform statistical comparisons
    comparison_results = {}
    
    for var in primary_vars:
        # ANOVA to test for overall differences
        groups = [df[df['diagnosis'] == group][var] for group in df['diagnosis'].unique()]
        groups = [group.dropna() for group in groups]  # Remove NaN values
        
        try:
            f_stat, p_value = stats.f_oneway(*groups)
            comparison_results[var] = {
                'F_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
            # If significant, perform post-hoc tests
            if p_value < 0.05:
                # Perform Tukey's HSD for post-hoc comparisons
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                posthoc = pairwise_tukeyhsd(
                    df[var].dropna(),
                    df.loc[df[var].notna(), 'diagnosis'],
                    alpha=0.05
                )
                comparison_results[var]['posthoc'] = posthoc
        except:
            comparison_results[var] = {
                'F_statistic': float('nan'),
                'p_value': float('nan'),
                'significant': False
            }
    
    # Apply FDR correction
    reject, corrected_pvals, _, _ = multipletests(p_value, alpha=0.05, method='fdr_bh')

    # Create formatted results
    table_data = []
    for i, var in enumerate(primary_vars):
        row = {'Variable': var}
        for group in df['diagnosis'].unique():
            m = group_stats.loc[group, (var, 'mean')]
            s = group_stats.loc[group, (var, 'std')]
            n = group_stats.loc[group, (var, 'count')]
            row[group] = f"{m:.2f} ± {s:.2f} (n={int(n)})"
        
        raw_p = p_values[i]
        corrected_p = corrected_pvals[i]
        row['Raw p-value'] = "p < 0.001" if raw_p < 0.001 else f"p = {raw_p:.3f}"
        row['FDR p-value'] = "p < 0.001" if corrected_p < 0.001 else f"p = {corrected_p:.3f}"
        row['FDR Significant'] = "Yes" if reject[i] else "No"
        table_data.append(row)
    
    # Final table
    outcome_table = pd.DataFrame(table_data)
    
    # Save if desired
    outcome_table.to_excel('primary_outcome_table_with_fdr.xlsx', index=False)
    
    # Create a formatted table for presentation
    table_data = []
    for var in primary_vars:
        row = {'Variable': var}
        for group in df['diagnosis'].unique():
            group_data = group_stats.loc[group, (var, 'mean')]
            group_std = group_stats.loc[group, (var, 'std')]
            group_n = group_stats.loc[group, (var, 'count')]
            row[f'{group}'] = f"{group_data:.2f} ± {group_std:.2f} (n={int(group_n)})"
        
        # Add statistical comparison results
        if comparison_results[var]['p_value'] < 0.001:
            row['p-value'] = "p < 0.001"
        else:
            row['p-value'] = f"p = {comparison_results[var]['p_value']:.3f}"
        
        row['Significant'] = "Yes" if comparison_results[var]['significant'] else "No"
        table_data.append(row)
    
    primary_outcome_table = pd.DataFrame(table_data)
    
    # Save table to Excel
    primary_outcome_table.to_excel('primary_outcome_table.xlsx', index=False)
    
    return primary_outcome_table, comparison_results
"""

# Function to create visualizations for quartile dispersions
def plot_quartile_dispersions(df):
    """
    Create visualizations for happy and sad quartile dispersions across diagnostic groups
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Happy quartile dispersion
    sns.boxplot(x='diagnosis', y='happy_quartile_dispersion', data=df, ax=axes[0])
    axes[0].set_title('Happy Quartile Dispersion by Diagnostic Group')
    axes[0].set_xlabel('Diagnostic Group')
    axes[0].set_ylabel('Happy Quartile Dispersion')
    
    # Sad quartile dispersion
    sns.boxplot(x='diagnosis', y='sad_quartile_dispersion', data=df, ax=axes[1])
    axes[1].set_title('Sad Quartile Dispersion by Diagnostic Group')
    axes[1].set_xlabel('Diagnostic Group')
    axes[1].set_ylabel('Sad Quartile Dispersion')
    
    plt.tight_layout()
    plt.savefig('quartile_dispersion_by_group.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create combined violin plots for better visualization
    plt.figure(figsize=(14, 8))
    
    plt.subplot(1, 2, 1)
    sns.violinplot(x='diagnosis', y='happy_quartile_dispersion', data=df, palette='Set2')
    plt.title('Happy Quartile Dispersion Distribution by Group')
    
    plt.subplot(1, 2, 2)
    sns.violinplot(x='diagnosis', y='sad_quartile_dispersion', data=df, palette='Set2')
    plt.title('Sad Quartile Dispersion Distribution by Group')
    
    plt.tight_layout()
    plt.savefig('quartile_dispersion_violins.png', dpi=300, bbox_inches='tight')

def main():
    # File path - replace with your actual file path
    file_path = '/Users/peterpressman/Desktop/Facex-1-27-25.xlsx'
    
   # Load and validate data
    print("Loading and validating data...")
    df_raw, numerical_vars, categorical_vars = load_and_validate_data(file_path)
    
    # Preprocess the data
    print("Preprocessing data...")
    df = preprocess_data(df_raw)
    
    # Print basic information about the dataset
    print("\nDataset Overview:")
    print(f"Total number of participants: {len(df)}")
    print("\nParticipants per group:")
    print(df['diagnosis'].value_counts())
    print(f"\nNumber of numerical variables: {len(numerical_vars)}")
    print(f"Number of categorical variables: {len(categorical_vars)}")
    
    # Calculate differences for numerical variables
    if numerical_vars:
        print("\nAnalyzing numerical variables...")
        num_results = calculate_numerical_differences(df, numerical_vars)
        
        # Create and save numerical heatmaps
        plot_heatmap(num_results['unadjusted'], 
                    num_results['effect_sizes'],
                    'Unadjusted Group Differences (Numerical Variables)')
        plt.savefig('numerical_unadjusted_heatmap.png', dpi=300, bbox_inches='tight')
        
        plot_heatmap(num_results['fwe_adjusted'],
                    num_results['effect_sizes'],
                    'FDR-Adjusted Group Differences (Numerical Variables)')
        plt.savefig('numerical_fwe_adjusted_heatmap.png', dpi=300, bbox_inches='tight')
    
    # Calculate differences for categorical variables
    if categorical_vars:
        print("\nAnalyzing categorical variables...")
        cat_results = calculate_categorical_differences(df, categorical_vars)
        
        # Only try to plot if we got results
        if cat_results is not None:
            # Create and save categorical heatmaps
            plot_heatmap(cat_results['unadjusted'],
                        cat_results['effect_sizes'],
                        'Unadjusted Group Differences (Categorical Variables)')
            plt.savefig('categorical_unadjusted_heatmap.png', dpi=300, bbox_inches='tight')
            
            plot_heatmap(cat_results['fwe_adjusted'],
                        cat_results['effect_sizes'],
                        'FDR-Adjusted Group Differences (Categorical Variables)')
            plt.savefig('categorical_fwe_adjusted_heatmap.png', dpi=300, bbox_inches='tight')
        else:
            print("No categorical results to plot - skipping categorical heatmaps")
    
    print("\nPerforming categorical analysis for most_frequent_emotion...")


    # Variables of interest
    variables_of_interest = [
        'emotional_diversity', 
        'happy_dispersion', 
        'happy_quartile_dispersion', 
        'sad_dispersion', 
        'sad_quartile_dispersion'
    ]

    # Group by diagnosis and calculate mean and std
    grouped_stats = df.groupby('diagnosis')[variables_of_interest].agg(['mean', 'std'])

    # Flatten the column names for clarity
    grouped_stats.columns = ['_'.join(col) for col in grouped_stats.columns]

    # Use tabulate to display it in a clean table format
    print(tabulate(grouped_stats, headers="keys", tablefmt="grid"))

    
    emotion_results = analyze_most_frequent_emotion(df)
    
    # Add results to Excel
    # Save emotion results to separate Excel file (modified to avoid append error)
    print("\nSaving emotion analysis results...")
    with pd.ExcelWriter('emotion_analysis_results.xlsx', engine='openpyxl') as writer:
        emotion_results['contingency'].to_excel(writer, sheet_name='Most_Frequent_Emotion_Counts')
        emotion_results['percentages'].to_excel(writer, sheet_name='Most_Frequent_Emotion_Pct')
        emotion_results['posthoc'].to_excel(writer, sheet_name='Most_Frequent_Emotion_Posthoc')

    # Save detailed results to Excel
    # Save detailed results to Excel
    print("\nSaving detailed results...")
    with pd.ExcelWriter('group_difference_results.xlsx', engine='openpyxl') as writer:
        if numerical_vars and num_results is not None:
            num_results['unadjusted'].to_excel(writer, sheet_name='Numerical_Unadjusted')
            num_results['fwe_adjusted'].to_excel(writer, sheet_name='Numerical_FWE_Adjusted')
            num_results['effect_sizes'].to_excel(writer, sheet_name='Numerical_Effect_Sizes')
        
        if categorical_vars and cat_results is not None:
            cat_results['unadjusted'].to_excel(writer, sheet_name='Categorical_Unadjusted')
            cat_results['fwe_adjusted'].to_excel(writer, sheet_name='Categorical_FWE_Adjusted')
            cat_results['effect_sizes'].to_excel(writer, sheet_name='Categorical_Effect_Sizes')
    
    print("\nAnalysis complete!")

    # Plot distributions for key variables
    print("\nPlotting variable distributions...")
    plot_variable_distributions(df, numerical_vars)


    # Primary outcome visualization
    print("\nPlotting emotional diversity across diagnostic groups...")
    plot_emotional_diversity(df)
    plot_emotional_diversity_violin(df)

    summary_table = df.groupby('diagnosis')['emotional_diversity'].agg(['mean', 'std', 'count'])
    print("\nSummary of Emotional Diversity by Diagnosis:")
    print(tabulate(summary_table, headers="keys", tablefmt="grid"))
    
    print("\nCreating primary outcome table...")
    primary_table, comparisons = create_primary_outcome_table(df)
    print("\nPrimary Outcome Variables:")
    print(tabulate(primary_table, headers="keys", tablefmt="grid"))
    
    # Add our new comprehensive correlation analysis
    print("\nPerforming comprehensive correlation analysis with diversity...")
    diversity_correlations = correlation_with_diversity_analysis(df)
    
    # Identify potential mediators
    print("\nIdentifying potential mediators...")
    mediators = identify_potential_mediators(diversity_correlations, primary_table)


    correlation_table = correlation_with_pvalues(df)
    print("\nCorrelation and significance with emotional diversity:")
    print(tabulate(correlation_table.head(10), headers="keys", tablefmt="grid"))

    summary_table = correlation_and_group_diff_summary(df)
    print(summary_table.head(10))

    # Add these lines to call the new functions
    print("\nCreating primary outcome table...")
    primary_table, comparisons = create_primary_outcome_table(df)
    print("\nPrimary Outcome Variables:")
    print(tabulate(primary_table, headers="keys", tablefmt="grid"))
    
    print("\nPlotting quartile dispersion comparisons...")
    plot_quartile_dispersions(df)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
