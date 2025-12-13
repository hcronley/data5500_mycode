import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import os

# =============================
# ENSURE OUTPUT DIRECTORY EXISTS
# =============================
os.makedirs('results/modeling', exist_ok=True)

# =============================
# PREDICTION VS ACTUAL PLOTS
# =============================
def plot_predictions_vs_actual(models, X_val, y_val, X_val_scaled):
    print("\n--- Creating Prediction vs Actual Plots ---")
    n_models = len(models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.ravel() if n_models > 1 else [axes]
    
    for idx, (name, model) in enumerate(models.items()):
        predictions = _get_predictions(model, name, X_val, X_val_scaled)
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        axes[idx].scatter(y_val, predictions, alpha=0.5, s=20)
        min_val, max_val = min(y_val.min(), predictions.min()), max(y_val.max(), predictions.max())
        axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        axes[idx].set_xlabel('Actual Values')
        axes[idx].set_ylabel('Predicted Values')
        axes[idx].set_title(f'{name}\nRMSE: {rmse:.2f}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/modeling/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/modeling/predictions_vs_actual.png")
    plt.close()

# ======================
# RESIDUAL PLOTS
# ======================
def plot_residuals(models, X_val, y_val, X_val_scaled):
    print("\n--- Creating Residual Plots ---")
    n_models = len(models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.ravel() if n_models > 1 else [axes]
    
    for idx, (name, model) in enumerate(models.items()):
        predictions = _get_predictions(model, name, X_val, X_val_scaled)
        residuals = y_val - predictions
        axes[idx].scatter(predictions, residuals, alpha=0.5, s=20)
        axes[idx].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[idx].set_xlabel('Predicted Values')
        axes[idx].set_ylabel('Residuals')
        axes[idx].set_title(f'{name} Residuals\nMean: {residuals.mean():.2f}, Std: {residuals.std():.2f}')
        axes[idx].grid(True, alpha=0.3)
    
    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/modeling/residual_plots.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/modeling/residual_plots.png")
    plt.close()

# ======================
# ERROR DISTRIBUTION
# ======================
def plot_error_distribution(models, X_val, y_val, X_val_scaled):
    print("\n--- Creating Error Distribution Plots ---")
    n_models = len(models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.ravel() if n_models > 1 else [axes]
    
    for idx, (name, model) in enumerate(models.items()):
        predictions = _get_predictions(model, name, X_val, X_val_scaled)
        errors = y_val - predictions
        axes[idx].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[idx].axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
        axes[idx].set_xlabel('Prediction Error')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{name} Error Distribution')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/modeling/error_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/modeling/error_distribution.png")
    plt.close()

# ======================
# METRICS HEATMAP
# ======================
def plot_metrics_heatmap(results):
    print("\n--- Creating Metrics Heatmap ---")
    metrics_df = pd.DataFrame(results).T
    metrics_normalized = metrics_df.copy()
    
    # Invert RMSE and MAE for better visualization
    metrics_normalized['RMSE'] = 1 - (metrics_df['RMSE'] - metrics_df['RMSE'].min()) / (metrics_df['RMSE'].max() - metrics_df['RMSE'].min())
    metrics_normalized['MAE'] = 1 - (metrics_df['MAE'] - metrics_df['MAE'].min()) / (metrics_df['MAE'].max() - metrics_df['MAE'].min())
    metrics_normalized['R2'] = (metrics_df['R2'] - metrics_df['R2'].min()) / (metrics_df['R2'].max() - metrics_df['R2'].min())
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(metrics_normalized, annot=metrics_df, fmt='.4f', cmap='RdYlGn', 
                center=0.5, linewidths=1, cbar_kws={'label': 'Normalized Score (Higher is Better)'})
    plt.title('Model Performance Heatmap\n(Annotations show actual values)', fontsize=14, pad=20)
    plt.ylabel('Model')
    plt.xlabel('Metric')
    plt.tight_layout()
    plt.savefig('results/modeling/metrics_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/modeling/metrics_heatmap.png")
    plt.close()

# ======================
# TRAIN VS VALIDATION COMPARISON
# ======================
def plot_train_val_comparison(models, X_train, y_train, X_val, y_val, X_train_scaled, X_val_scaled):
    print("\n--- Creating Train vs Validation Comparison ---")
    train_rmse = {}
    val_rmse = {}
    
    for name, model in models.items():
        train_pred = _get_predictions(model, name, X_train, X_train_scaled)
        val_pred = _get_predictions(model, name, X_val, X_val_scaled)
        train_rmse[name] = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse[name] = np.sqrt(mean_squared_error(y_val, val_pred))
    
    comparison = pd.DataFrame({'Training RMSE': train_rmse, 'Validation RMSE': val_rmse})
    comparison['Overfit Gap'] = comparison['Validation RMSE'] - comparison['Training RMSE']
    comparison = comparison.sort_values('Validation RMSE')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    comparison[['Training RMSE', 'Validation RMSE']].plot(kind='barh', ax=axes[0], color=['steelblue', 'coral'])
    axes[0].set_xlabel('RMSE')
    axes[0].set_title('Training vs Validation RMSE')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    comparison['Overfit Gap'].plot(kind='barh', ax=axes[1], color='darkred')
    axes[1].set_xlabel('Overfit Gap (Val RMSE - Train RMSE)')
    axes[1].set_title('Overfitting Analysis\n(Lower is Better)')
    axes[1].axvline(x=0, color='black', linestyle='--', lw=1)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/modeling/train_val_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/modeling/train_val_comparison.png")
    print("\nOverfitting Analysis:")
    print(comparison[['Overfit Gap']].to_string())
    plt.close()

# ======================
# FEATURE IMPORTANCE COMPARISON
# ======================
def plot_feature_importance_comparison(models, feature_names):
    print("\n--- Creating Feature Importance Comparison ---")
    
    if 'Random Forest' not in models or 'XGBoost' not in models:
        print("Skipping: Need both Random Forest and XGBoost")
        return
    
    rf_imp = pd.DataFrame({'feature': feature_names, 'RF_importance': models['Random Forest'].feature_importances_}).sort_values('RF_importance', ascending=False)
    xgb_imp = pd.DataFrame({'feature': feature_names, 'XGB_importance': models['XGBoost'].feature_importances_}).sort_values('XGB_importance', ascending=False)
    
    merged = rf_imp.merge(xgb_imp, on='feature')
    top_features = list(set(rf_imp.head(15)['feature']) | set(xgb_imp.head(15)['feature']))
    merged_top = merged[merged['feature'].isin(top_features)]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].scatter(merged['RF_importance'], merged['XGB_importance'], alpha=0.6)
    for idx, row in merged_top.iterrows():
        axes[0].annotate(row['feature'], (row['RF_importance'], row['XGB_importance']), fontsize=8, alpha=0.7)
    max_imp = max(merged['RF_importance'].max(), merged['XGB_importance'].max())
    axes[0].plot([0, max_imp], [0, max_imp], 'r--', alpha=0.5, label='Perfect Agreement')
    axes[0].set_xlabel('Random Forest Importance')
    axes[0].set_ylabel('XGBoost Importance')
    axes[0].set_title('Feature Importance Comparison\n(Top features labeled)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    correlation = merged['RF_importance'].corr(merged['XGB_importance'])
    top_10_features = merged.nlargest(10, 'RF_importance')['feature'].tolist()
    comparison_data = merged[merged['feature'].isin(top_10_features)].set_index('feature')
    comparison_data[['RF_importance', 'XGB_importance']].plot(kind='barh', ax=axes[1])
    axes[1].set_xlabel('Importance')
    axes[1].set_title(f'Top 10 Features Side-by-Side\nCorrelation: {correlation:.3f}')
    axes[1].legend(['Random Forest', 'XGBoost'])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/modeling/feature_importance_detailed.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/modeling/feature_importance_detailed.png")
    print(f"\nFeature importance correlation: {correlation:.3f}")
    plt.close()

# ======================
# SUMMARY DASHBOARD
# ======================
def create_summary_dashboard(results):
    print("\n--- Creating Summary Dashboard ---")
    df = pd.DataFrame(results).T.sort_values('RMSE')
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])
    df['RMSE'].plot(kind='barh', ax=ax1, color='steelblue')
    ax1.set_xlabel('RMSE (Lower is Better)')
    ax1.set_title('Model RMSE Comparison', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(df['MAE'], df['RMSE'], s=100, alpha=0.6)
    for idx, name in enumerate(df.index):
        ax2.annotate(name, (df.loc[name, 'MAE'], df.loc[name, 'RMSE']), fontsize=8, alpha=0.7)
    ax2.set_xlabel('MAE')
    ax2.set_ylabel('RMSE')
    ax2.set_title('MAE vs RMSE')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 1])
    df['R2'].plot(kind='bar', ax=ax3, color='seagreen')
    ax3.set_ylabel('R² Score (Higher is Better)')
    ax3.set_title('R² Score Comparison')
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    best_model = df['RMSE'].idxmin()
    best_rmse, best_r2, best_mae = df.loc[best_model, ['RMSE', 'R2', 'MAE']]
    text = f"🏆 BEST MODEL\n\nModel: {best_model}\n\nMetrics:\n• RMSE: {best_rmse:.4f}\n• MAE: {best_mae:.4f}\n• R²: {best_r2:.4f}\n\nTotal Models: {len(df)}"
    ax4.text(0.1, 0.5, text, fontsize=12, verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('tight')
    ax5.axis('off')
    table_data = df.round(4).reset_index()
    table_data.columns = ['Model', 'RMSE', 'MAE', 'R²']
    table = ax5.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    best_idx = table_data[table_data['Model'] == best_model].index[0] + 1
    for i in range(4):
        table[(best_idx, i)].set_facecolor('lightgreen')
    
    plt.suptitle('MODEL PERFORMANCE SUMMARY DASHBOARD', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('results/modeling/summary_dashboard.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/modeling/summary_dashboard.png")
    plt.close()

# ======================
# HELPER FUNCTION: GET PREDICTIONS
# ======================
def _get_predictions(model, name, X, X_scaled):
    if 'NN' in name or 'Ridge' in name:
        return model.predict(X_scaled, verbose=0).flatten() if 'NN' in name else model.predict(X_scaled)
    else:
        return model.predict(X)

# ======================
# MASTER FUNCTION
# ======================
def create_all_visualizations(models, results, X_train, y_train, X_val, y_val, X_train_scaled, X_val_scaled, feature_names):
    print("\nCREATING ALL VISUALIZATIONS")
    
    plot_predictions_vs_actual(models, X_val, y_val, X_val_scaled)
    plot_residuals(models, X_val, y_val, X_val_scaled)
    plot_error_distribution(models, X_val, y_val, X_val_scaled)
    plot_metrics_heatmap(results)
    plot_train_val_comparison(models, X_train, y_train, X_val, y_val, X_train_scaled, X_val_scaled)
    plot_feature_importance_comparison(models, feature_names)
    create_summary_dashboard(results)
    
    print("\nAll visualizations generated and saved in 'results/modeling/' folder.\n")