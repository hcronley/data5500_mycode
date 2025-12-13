#!/home/ubuntu/pycaret_env/bin/python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pycaret.regression import *
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
import logging
from visualizations import create_all_visualizations

# Suppress font warnings
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

def setup_output_directories():
    """Create directories for organized output."""
    import os
    
    directories = [
        'results',
        'results/eda',
        'results/modeling',
        'results/models'  # For saving trained models if needed
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Output directories created")

# Keras imports
try:
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    from tensorflow.keras.models import Model
    KERAS_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow/Keras not available. Neural networks will be skipped.")
    KERAS_AVAILABLE = False

# =============================
# NEURAL NETWORK CLASSES (OOP)
# =============================
if KERAS_AVAILABLE:
    class BaseNeuralNetwork(Model):
        """
        Base class for neural networks demonstrating OOP inheritance.
        All custom neural networks inherit from this class.
        """
        def __init__(self, name="BaseNN"):
            super(BaseNeuralNetwork, self).__init__(name=name)
            self.model_name = name
        
        def get_config(self):
            return {"name": self.model_name}
        
        def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
            """Common training method for all neural networks."""
            self.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            history = self.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=0
            )
            
            return history
    
    
    class ShallowNeuralNetwork(BaseNeuralNetwork):
        """
        Shallow Neural Network (2 hidden layers).
        Good for simpler patterns and faster training.
        """
        def __init__(self, input_dim):
            super(ShallowNeuralNetwork, self).__init__(name="ShallowNN")
            self.dense1 = layers.Dense(64, activation='relu')
            self.dropout1 = layers.Dropout(0.2)
            self.dense2 = layers.Dense(32, activation='relu')
            self.output_layer = layers.Dense(1)
        
        def call(self, inputs):
            x = self.dense1(inputs)
            x = self.dropout1(x)
            x = self.dense2(x)
            return self.output_layer(x)
    
    
    class DeepNeuralNetwork(BaseNeuralNetwork):
        """
        Deep Neural Network (4 hidden layers).
        Can capture more complex patterns but requires more data.
        """
        def __init__(self, input_dim):
            super(DeepNeuralNetwork, self).__init__(name="DeepNN")
            self.dense1 = layers.Dense(128, activation='relu')
            self.dropout1 = layers.Dropout(0.3)
            self.dense2 = layers.Dense(64, activation='relu')
            self.dropout2 = layers.Dropout(0.2)
            self.dense3 = layers.Dense(32, activation='relu')
            self.dropout3 = layers.Dropout(0.2)
            self.dense4 = layers.Dense(16, activation='relu')
            self.output_layer = layers.Dense(1)
        
        def call(self, inputs):
            x = self.dense1(inputs)
            x = self.dropout1(x)
            x = self.dense2(x)
            x = self.dropout2(x)
            x = self.dense3(x)
            x = self.dropout3(x)
            x = self.dense4(x)
            return self.output_layer(x)
    
    
    class WideNeuralNetwork(BaseNeuralNetwork):
        """
        Wide Neural Network (fewer layers but more neurons).
        Good for learning many simple patterns simultaneously.
        """
        def __init__(self, input_dim):
            super(WideNeuralNetwork, self).__init__(name="WideNN")
            self.dense1 = layers.Dense(256, activation='relu')
            self.dropout1 = layers.Dropout(0.3)
            self.dense2 = layers.Dense(128, activation='relu')
            self.dropout2 = layers.Dropout(0.2)
            self.output_layer = layers.Dense(1)
        
        def call(self, inputs):
            x = self.dense1(inputs)
            x = self.dropout1(x)
            x = self.dense2(x)
            x = self.dropout2(x)
            return self.output_layer(x)

# =============
# DATA LOADING
# =============
def load_data(train_path, test_path, sample_path):
    print("\n Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    
    print(f"train shape: {train.shape}")
    print(f"test shape: {test.shape}")
    print(f"sample shape: {sample.shape}")
    
    return train, test, sample

# ====================
# RELABELING COLUMNS
# ====================
def relabel_columns_from_reference(X, X_test):
    """
    Relabel columns deterministically to f_0, f_1, ...
    using training data as the reference.
    """
    new_cols = [f"f_{i}" for i in range(X.shape[1])]
    X = X.copy()
    X_test = X_test.copy()
    X.columns = new_cols
    X_test.columns = new_cols
    return X, X_test

# ====================
# PREPROCESSING
# ====================
def preprocess_data(train, test, target='CORRUCYSTIC_DENSITY'):
    """
    Preprocess train and test data:
    - Handle missing values
    - Encode categorical variables
    - Scale features
    """
    print("\n --- Preprocessing Data ---")

    # Separate features and target
    X = train.drop(columns=[target])
    y = train[target]
    X_test = test.copy()

    # Remove LOCAL_IDENTIFIER (it's just an index)
    if 'LOCAL_IDENTIFIER' in X.columns:
        X = X.drop(columns=['LOCAL_IDENTIFIER'])
    if 'LOCAL_IDENTIFIER' in X_test.columns:
        X_test = X_test.drop(columns=['LOCAL_IDENTIFIER'])
    
    print("\nOriginal training samples: ", len(X))
    print("Target missing values: ", y.isnull().sum())
    
    # Remove rows where target is missing
    valid_idx = ~y.isnull()
    X = X[valid_idx]
    y = y[valid_idx]
    print("After removing missing targets: ", len(X))

    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print("\nCategorical features: ", len(categorical_cols))
    print("Numeric features: ", len(numeric_cols))

    # Handle missing values in numeric features (median imputation)
    print("\nImputing Missing Values")
    numeric_imputer = SimpleImputer(strategy='median')
    X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
    X_test[numeric_cols] = numeric_imputer.transform(X_test[numeric_cols])
    print("Numeric features imputed with median")
    
    # Drop categorical features (weak correlation anyway)
    if len(categorical_cols) > 0:
        print(f"\nDropping {len(categorical_cols)} categorical features")
        X = X.drop(columns=categorical_cols)
        X_test = X_test.drop(columns=categorical_cols)
        print("Categorical features removed")

    # Relabel columns (competition uses scrambled feature names)
    print("\nRelabeling feature columns")
    X, X_test = relabel_columns_from_reference(X, X_test)
    print("Columns relabeled for model compatibility")

    # Feature scaling (for neural network)
    print("\nScaling Features")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    print("Features scaled (StandardScaler)")
    
    print("\nFinal preprocessed shape: ", X.shape)
    print("Final test shape: ", X_test.shape)
    
    return X, y, X_test, X_scaled, X_test_scaled, scaler

# =======================
# TRAIN/VALIDATION SPLIT
# =======================
def create_train_val_split(X, y, X_scaled, test_size=0.2, random_state=42):
    """Split data into training and validation sets."""
    print("\n --- Split into Training and Validation Sets ---")

    # Split unscaled data (for tree-based models)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Split scaled data (for neural network)
    X_train_scaled, X_val_scaled, _, _ = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Split ratio: {100*(1-test_size):.0f}/{100*test_size:.0f}")
    
    return X_train, X_val, y_train, y_val, X_train_scaled, X_val_scaled

# ====================
# MODEL TRAINING
# ====================
def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest model."""
    print("\n--- Training Random Forest ---")
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Val RMSE: {val_rmse:.4f}")
    print(f"Val MAE: {val_mae:.4f}")
    print(f"Val R²: {val_r2:.4f}")
    
    return model, val_rmse, val_mae, val_r2

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model."""
    print("\n--- Training XGBoost ---")

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Val RMSE: {val_rmse:.4f}")
    print(f"Val MAE: {val_mae:.4f}")
    print(f"Val R²: {val_r2:.4f}")
    
    return model, val_rmse, val_mae, val_r2

def train_neural_network(model, X_train_scaled, y_train, X_val_scaled, y_val, epochs=50):
    """Train a custom neural network model."""
    print(f"\n--- Training {model.model_name} ---")
    
    if not KERAS_AVAILABLE:
        print("Keras not available, skipping...")
        return None, None, None, None
    
    # Build model by calling it once
    _ = model(X_train_scaled[:1])
    
    # Train using the custom training method
    history = model.train_model(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        epochs=epochs,
        batch_size=32
    )
    
    # Predictions
    train_pred = model.predict(X_train_scaled, verbose=0).flatten()
    val_pred = model.predict(X_val_scaled, verbose=0).flatten()
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Val RMSE: {val_rmse:.4f}")
    print(f"Val MAE: {val_mae:.4f}")
    print(f"Val R²: {val_r2:.4f}")
    
    return model, val_rmse, val_mae, val_r2

def train_ridge_regression(X_train_scaled, y_train, X_val_scaled, y_val):
    """Train Ridge Regression model."""
    print("\n--- Training Ridge Regression ---")
    
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Val RMSE: {val_rmse:.4f}")
    print(f"Val MAE: {val_mae:.4f}")
    print(f"Val R²: {val_r2:.4f}")
    
    return model, val_rmse, val_mae, val_r2

def train_gradient_boosting(X_train, y_train, X_val, y_val):
    """Train Gradient Boosting model."""
    print("\n--- Training Gradient Boosting ---")
    
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Val RMSE: {val_rmse:.4f}")
    print(f"Val MAE: {val_mae:.4f}")
    print(f"Val R²: {val_r2:.4f}")
    
    return model, val_rmse, val_mae, val_r2

# ============================
# FEATURE IMPORTANCE ANALYSIS
# ============================
def analyze_feature_importance(models, feature_names, top_n=20):
    """Analyze and visualize feature importance from tree-based models."""
    print("\n --- Feature Importance Analysis ---")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Random Forest importance
    if 'Random Forest' in models:
        rf_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': models['Random Forest'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nRandom Forest Top 15 Features")
        print(rf_importance.head(15))
        
        rf_importance.head(top_n).plot(
            x='feature', y='importance', kind='barh', ax=axes[0], legend=False
        )
        axes[0].set_title('Random Forest Feature Importance')
        axes[0].set_xlabel('Importance')
        axes[0].invert_yaxis()
    
    # XGBoost importance
    if 'XGBoost' in models:
        xgb_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': models['XGBoost'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nXGBoost Top 15 Features")
        print(xgb_importance.head(15))
        
        xgb_importance.head(top_n).plot(
            x='feature', y='importance', kind='barh', ax=axes[1], legend=False
        )
        axes[1].set_title('XGBoost Feature Importance')
        axes[1].set_xlabel('Importance')
        axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('results/modeling/feature_importance.png', dpi=300, bbox_inches='tight')
    print("\nSaved: results/modeling/feature_importance.png")
    plt.close()

# ====================
# MODEL COMPARISON
# ====================
def compare_models(results):
    """Compare all models and create visualization."""
    print("\n --- Model Comparison ---")
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.sort_values('RMSE')
    
    print("\nModel Performance Comparison")
    print(comparison_df.to_string())
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # RMSE comparison
    comparison_df['RMSE'].plot(kind='barh', ax=axes[0], color='steelblue')
    axes[0].set_title('RMSE Comparison (Lower is Better)')
    axes[0].set_xlabel('RMSE')
    axes[0].invert_yaxis()
    
    # MAE comparison
    comparison_df['MAE'].plot(kind='barh', ax=axes[1], color='coral')
    axes[1].set_title('MAE Comparison (Lower is Better)')
    axes[1].set_xlabel('MAE')
    axes[1].invert_yaxis()
    
    # R² comparison
    comparison_df['R2'].plot(kind='barh', ax=axes[2], color='seagreen')
    axes[2].set_title('R² Comparison (Higher is Better)')
    axes[2].set_xlabel('R² Score')
    axes[2].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('results/modeling/model_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved: results/modeling/model_comparison.png")
    plt.close()
    
    # Print best model
    best_model = comparison_df['RMSE'].idxmin()
    print(f"\n  Best Model: {best_model}")
    print(f"   RMSE: {comparison_df.loc[best_model, 'RMSE']:.4f}")
    print(f"   R²: {comparison_df.loc[best_model, 'R2']:.4f}")
    
    return comparison_df

# ========================================
# NEURAL NETWORK ARCHITECTURE COMPARISON
# ========================================
def compare_neural_networks(results):
    """Create separate comparison for neural network architectures."""
    print("\n --- Neural Network Comparison ---")
    
    # Filter only neural network results
    nn_results = {k: v for k, v in results.items() if 'NN' in k or 'Neural' in k}
    
    if len(nn_results) == 0:
        print("No neural networks to compare")
        return
    
    nn_df = pd.DataFrame(nn_results).T
    nn_df = nn_df.sort_values('RMSE')
    
    print("\nNeural Network Performance")
    print(nn_df.to_string())
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    nn_df['RMSE'].plot(kind='bar', ax=axes[0], color='purple', alpha=0.7)
    axes[0].set_title('Neural Network RMSE Comparison')
    axes[0].set_ylabel('RMSE (Lower is Better)')
    axes[0].set_xlabel('Architecture')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    nn_df['R2'].plot(kind='bar', ax=axes[1], color='teal', alpha=0.7)
    axes[1].set_title('Neural Network R² Comparison')
    axes[1].set_ylabel('R² Score (Higher is Better)')
    axes[1].set_xlabel('Architecture')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('results/modeling/neural_network_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved: results/modeling/neural_network_comparison.png")
    plt.close()
    
    # Best NN
    best_nn = nn_df['RMSE'].idxmin()
    print(f"\nBest Neural Network: {best_nn}")
    print(f"   RMSE: {nn_df.loc[best_nn, 'RMSE']:.4f}")

# ====================
# MAIN PIPELINE
# ====================
def run_pipeline(train_path, test_path, sample_path, target='CORRUCYSTIC_DENSITY'):
    """Run complete modeling pipeline."""
    print("\nMACHINE LEARNING PIPELINE - CORRUCYSTIC_DENSITY PREDICTION")
    
    setup_output_directories()

    # 1. Load data
    train, test, sample = load_data(train_path, test_path, sample_path)
    
    # 2. Preprocess
    X, y, X_test, X_scaled, X_test_scaled, scaler = preprocess_data(train, test, target)
    
    # 3. Train/val split
    X_train, X_val, y_train, y_val, X_train_scaled, X_val_scaled = create_train_val_split(
        X, y, X_scaled
    )
    
    # 4. Train models
    print("\nTRAINING MODELS")
    
    models = {}
    results = {}
    
    # Random Forest (REQUIRED)
    rf_model, rf_rmse, rf_mae, rf_r2 = train_random_forest(X_train, y_train, X_val, y_val)
    models['Random Forest'] = rf_model
    results['Random Forest'] = {'RMSE': rf_rmse, 'MAE': rf_mae, 'R2': rf_r2}
    
    # XGBoost (REQUIRED)
    xgb_model, xgb_rmse, xgb_mae, xgb_r2 = train_xgboost(X_train, y_train, X_val, y_val)
    models['XGBoost'] = xgb_model
    results['XGBoost'] = {'RMSE': xgb_rmse, 'MAE': xgb_mae, 'R2': xgb_r2}
    
    # Neural Networks (REQUIRED - Multiple Architectures with OOP)
    if KERAS_AVAILABLE:
        input_dim = X_train_scaled.shape[1]
        
        # Shallow NN
        shallow_nn = ShallowNeuralNetwork(input_dim)
        shallow_model, shallow_rmse, shallow_mae, shallow_r2 = train_neural_network(
            shallow_nn, X_train_scaled, y_train, X_val_scaled, y_val
        )
        if shallow_model is not None:
            models['Shallow NN'] = shallow_model
            results['Shallow NN'] = {'RMSE': shallow_rmse, 'MAE': shallow_mae, 'R2': shallow_r2}
        
        # Deep NN
        deep_nn = DeepNeuralNetwork(input_dim)
        deep_model, deep_rmse, deep_mae, deep_r2 = train_neural_network(
            deep_nn, X_train_scaled, y_train, X_val_scaled, y_val
        )
        if deep_model is not None:
            models['Deep NN'] = deep_model
            results['Deep NN'] = {'RMSE': deep_rmse, 'MAE': deep_mae, 'R2': deep_r2}
        
        # Wide NN
        wide_nn = WideNeuralNetwork(input_dim)
        wide_model, wide_rmse, wide_mae, wide_r2 = train_neural_network(
            wide_nn, X_train_scaled, y_train, X_val_scaled, y_val
        )
        if wide_model is not None:
            models['Wide NN'] = wide_model
            results['Wide NN'] = {'RMSE': wide_rmse, 'MAE': wide_mae, 'R2': wide_r2}
    
    # Additional models
    ridge_model, ridge_rmse, ridge_mae, ridge_r2 = train_ridge_regression(
        X_train_scaled, y_train, X_val_scaled, y_val
    )
    models['Ridge Regression'] = ridge_model
    results['Ridge Regression'] = {'RMSE': ridge_rmse, 'MAE': ridge_mae, 'R2': ridge_r2}
    
    gb_model, gb_rmse, gb_mae, gb_r2 = train_gradient_boosting(
        X_train, y_train, X_val, y_val
    )
    models['Gradient Boosting'] = gb_model
    results['Gradient Boosting'] = {'RMSE': gb_rmse, 'MAE': gb_mae, 'R2': gb_r2}
    
    # 5. Feature importance (REQUIRED)
    analyze_feature_importance(models, X_train.columns.tolist())
    
    # 6. Compare models
    comparison_df = compare_models(results)

    # 7. Compare neural network architectures
    compare_neural_networks(results)

    # 8. Advanced visualizations
    create_all_visualizations(
        models, results,
        X_train, y_train, X_val, y_val,
        X_train_scaled, X_val_scaled,
        X_train.columns.tolist()
    )
    
    print("\nPIPELINE COMPLETE!")
    print("\nGenerated files:")
    print("  • feature_importance.png - Feature importance from RF and XGBoost")
    print("  • model_comparison.png - Performance comparison of all models")
    print("  • neural_network_comparison.png - Neural network architecture comparison")
    
    return models, results, comparison_df, X_test, X_test_scaled

# ============================
# COMPETITION SUBMISSION
# ============================
def generate_submission(best_model, X_test, sample_path, output_file='results/submission.csv'):
    """Generate Kaggle submission file."""
    sample = pd.read_csv(sample_path)
    predictions = best_model.predict(X_test)
    
    submission = pd.DataFrame({
        'LOCAL_IDENTIFIER': sample['LOCAL_IDENTIFIER'],
        'CORRUCYSTIC_DENSITY': predictions
    })
    
    submission.to_csv(output_file, index=False)
    print(f"\nSubmission saved to {output_file}")
    return submission

# ================
# MAIN EXECUTION
# ================
if __name__ == "__main__":
    # Run pipeline
    models, results, comparison_df, X_test, X_test_scaled = run_pipeline(
        train_path="/home/ubuntu/pycaret_env/final_project/data/MiNDAT.csv",
        test_path="/home/ubuntu/pycaret_env/final_project/data/MiNDAT_UNK.csv",
        sample_path="/home/ubuntu/pycaret_env/final_project/data/SPECIMEN.csv"
    )

    # Pick best model based on lowest RMSE
    best_model_name = comparison_df['RMSE'].idxmin()
    best_model = models[best_model_name]
    print(f"\nGenerating submission with: {best_model_name}")

    # Generate submission CSV in the results folder
    submission = generate_submission(
        best_model=best_model,
        X_test=X_test,
        sample_path="/home/ubuntu/pycaret_env/final_project/data/SPECIMEN.csv",
        output_file='results/submission.csv'
    )

