#!/home/ubuntu/pycaret_env/bin/python
import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import probplot
import sklearn 
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.regression import *
import warnings
import logging

# Suppress font warnings
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

def load_data(train_path, test_path, sample_path):
    print("\n Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    
    print(f"train shape: {train.shape}")
    print(f"test shape: {test.shape}")
    print(f"sample shape: {sample.shape}")
    
    return train, test, sample

def basic_info(train):
    print("\n --- Basic Dataset Info ---")

    print("\nTrain Data Info")
    print(train.info())

    print("\nFirst few rows")
    print(train.head())

    print("\nLast few rows")
    print(train.tail())

    print("\nData types summary")
    print(train.dtypes.value_counts())

def analyze_target(train, target):
    """Analyze and visualize the target variable distribution."""
    print("\n --- Target Variable Analysis ---")

    if target in train.columns:
        print("\n", target, "Statistics")
        print(train[target].describe())

        print("\nSkewness: ", train[target].skew())
        print("Kurtosis: ", train[target].kurtosis())

        # Visualize target distribution
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Histogram
        axes[0].hist(train[target], bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel(target)
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'{target} Distribution')
        axes[0].axvline(train[target].mean(), color='red', linestyle='--', label='Mean')
        axes[0].axvline(train[target].median(), color='green', linestyle='--', label='Median')
        axes[0].legend()
        

        # Box plot
        sns.boxplot(y=train[target], ax=axes[1])
        axes[1].set_ylabel(target)
        axes[1].set_title(f'{target} Box Plot')
        
        # Q-Q plot
        probplot(train[target], dist="norm", plot=axes[2])
        axes[2].set_title(f'{target} Q-Q Plot')
        
        plt.tight_layout()
        plt.savefig('results/eda/target_distribution.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: results/eda/target_distribution.png")
        plt.close()

def analyze_missing_values(train, test):
    """Identify and visualize missing values in train and test datasets."""
    print("\n --- Missing Value Analysis ---")

    train_missing = train.isnull().sum()
    train_missing_pct = (train_missing / len(train)) * 100
    missing_df = pd.DataFrame({
        'Missing_Count': train_missing,
        'Percentage': train_missing_pct
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    print("\nTraining missing: \n", missing_df)

    # Visualize missing values
    plt.figure(figsize=(12, 6))
    missing_df['Percentage'].plot(kind='barh')
    plt.xlabel('Percentage Missing (%)')
    plt.title('Missing Values by Feature')

    plt.tight_layout()
    plt.savefig('results/eda/missing_values.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: results/eda/missing_values.png")
    plt.close()

    # Check test set
    test_missing = test.isnull().sum()
    test_missing = test_missing[test_missing > 0]
    print("\nTest missing: \n", test_missing)

def analyze_feature_types(train, target):
    """Categorize features as numeric or categorical and display statistics."""
    print("\n --- Feature Types & Stats ---")

    # Identify feature types
    numeric_features = train.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_features:
        numeric_features.remove(target)

    categorical_features = train.select_dtypes(include=['object']).columns.tolist()

    print("\nNumeric features: ", len(numeric_features))
    print("Categorical features: ", len(categorical_features))
    print("Total features: ", len(categorical_features) + len(numeric_features))

    print("\nNumeric Features")
    print(numeric_features[:10], "..." if len(numeric_features) > 10 else "")

    if categorical_features:
        print("\nCategorical Features")
        print(categorical_features)
        
        print("\nCategorical Feature Cardinality")
        for col in categorical_features:
            print(f"{col}: {train[col].nunique()} unique values")

    # Statistical summary of numeric features
    print("\nNumeric Features Summary")
    print(train[numeric_features].describe().T)

    return numeric_features, categorical_features

def analyze_correlations(train, target, numeric_features):
    """Calculate and visualize correlations between features and target."""
    print("\n --- Correlation Analysis ---")

    if target in train.columns and len(numeric_features) > 0:
        # Correlation with target
        correlations = train[numeric_features + [target]].corr()[target].sort_values(ascending=False)
        print("\nTop 15 Features Correlated with Target")
        print(correlations.head(15))
        
        print("\nBottom 15 Features Correlated with Target")
        print(correlations.tail(15))
        
        # Visualize top correlations
        top_n = min(20, len(correlations) - 1)
        top_corr = correlations[1:top_n+1]  # Exclude target itself
        
        plt.figure(figsize=(10, 8))
        top_corr.plot(kind='barh')
        plt.xlabel('Correlation with Target')
        plt.title(f'Top {top_n} Features Correlated with {target}')
        plt.tight_layout()
        plt.savefig('target_correlations.png', dpi=300, bbox_inches='tight')
        print("\nSaved: target_correlations.png")
        plt.close()

        # Full correlation heatmap (for top features only to keep readable)
        top_features = correlations.abs().sort_values(ascending=False).head(16).index.tolist()
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(train[top_features].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)

        plt.tight_layout()
        plt.savefig('results/eda/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: results/eda/correlation_heatmap.png")
        plt.close()

    return correlations

def analyze_distributions(train, numeric_features):
    """Analyze and visualize the distribution and skewness of numeric features."""
    print("\n --- Distribution Analysis ---")

    # Analyze skewness
    if len(numeric_features) > 0:
        skewness = train[numeric_features].skew().sort_values(ascending=False)
        print("\nMost Skewed Features")
        print(skewness.head(10))
        
        # Identify highly skewed features (absolute skew > 1)
        highly_skewed = skewness[abs(skewness) > 1]
        print(f"\nFeatures with |skewness| > 1: {len(highly_skewed)}")
        
        # Plot distributions of highly skewed features (first 9)
        if len(highly_skewed) > 0:
            n_plots = min(9, len(highly_skewed))
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            axes = axes.ravel()
            
            for i, col in enumerate(highly_skewed.head(n_plots).index):
                axes[i].hist(train[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
                axes[i].set_title(f'{col}\nSkew: {skewness[col]:.2f}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
            
            # Hide unused subplots
            for j in range(i+1, 9):
                axes[j].axis('off')

            plt.tight_layout()
            plt.savefig('results/eda/skewed_distributions.png', dpi=300, bbox_inches='tight')
            print(f"\nSaved: results/eda/skewed_distributions.png")
            plt.close()
            

    return

def analyze_outliers(train, numeric_features):
    """Detect and visualize outliers in numeric features using IQR method."""
    print("\n --- Detection of Outliers ---")

    def count_outliers(series):
        """Count outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return ((series < lower) | (series > upper)).sum()

    if len(numeric_features) > 0:
        outlier_counts = {}
        for col in numeric_features:
            outlier_counts[col] = count_outliers(train[col])
    
    outlier_df = pd.DataFrame.from_dict(outlier_counts, orient='index', columns=['Outlier_Count'])
    outlier_df['Outlier_Percentage'] = (outlier_df['Outlier_Count'] / len(train)) * 100
    outlier_df = outlier_df.sort_values('Outlier_Count', ascending=False)
    
    print("\nFeatures with Most Outliers")
    print(outlier_df.head(15))
    
    # Visualize features with most outliers
    top_outlier_features = outlier_df.head(9).index.tolist()
    
    if len(top_outlier_features) > 0:
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, col in enumerate(top_outlier_features):
            axes[i].boxplot(train[col].dropna())
            axes[i].set_title(f'{col}\n{outlier_df.loc[col, "Outlier_Count"]:.0f} outliers ({outlier_df.loc[col, "Outlier_Percentage"]:.1f}%)')
            axes[i].set_ylabel(col)
        
        # Hide unused subplots
        for j in range(len(top_outlier_features), 9):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig('results/eda/outlier_boxplots.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: results/eda/outlier_boxplots.png")
        plt.close()

        return

def analyze_feature_relationships(train, target, correlations, numeric_features):
    """Create scatter plots showing relationships between top features and target."""
    print("\n --- Feature Relationships w/ Target ---")

    if target in train.columns and len(numeric_features) > 0:
        # Select top correlated features for scatter plots
        top_features_for_plot = correlations.abs().sort_values(ascending=False)[1:10].index.tolist()
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, col in enumerate(top_features_for_plot):
            axes[i].scatter(train[col], train[target], alpha=0.3, s=1)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel(target)
            axes[i].set_title(f'{col} vs {target}\nCorr: {correlations[col]:.3f}')

        plt.tight_layout()
        plt.savefig('results/eda/feature_target_relationships.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: results/eda/feature_target_relationships.png")
        plt.close()

def run_eda(train_path, test_path, sample_path, target='CORRUCYSTIC_DENSITY'):
    """Run complete EDA pipeline."""
    print("CORRUCYSTIC_DENSITY PREDICTION - EXPLORATORY DATA ANALYSIS")
    
    # Load data
    train, test, sample = load_data(train_path, test_path, sample_path)
    
    # Run analysis sections
    basic_info(train)
    analyze_target(train, target)
    analyze_missing_values(train, test)
    numeric_features, categorical_features = analyze_feature_types(train, target)
    correlations = analyze_correlations(train, target, numeric_features)
    analyze_distributions(train, numeric_features)
    analyze_outliers(train, numeric_features)
    analyze_feature_relationships(train, target, correlations, numeric_features)

    print("EDA COMPLETE!")
    
    return train, test, sample, numeric_features, categorical_features

# =======================
# MAIN EXECUTION
# =======================
if __name__ == "__main__":
    train, test, sample, numeric_features, categorical_features = run_eda(
        train_path="/home/ubuntu/pycaret_env/final_project/data/MiNDAT.csv",
        test_path="/home/ubuntu/pycaret_env/final_project/data/MiNDAT_UNK.csv",
        sample_path="/home/ubuntu/pycaret_env/final_project/data/SPECIMEN.csv"
    )

