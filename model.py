import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss, f1_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# 1. Load Data
train_path = r"c:\aty\projects\dsv\preprocessed_data_train.parquet"
test_path = r"c:\aty\projects\dsv\preprocessed_data_test.parquet"

df_train_full = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

###################################################################################################################
# Separate Features, Targets, and Groups
target_col = "label"
group_col = "page_id"

# Ensure target and group columns are excluded from training features
feature_cols = [col for col in df_train_full.columns if col not in ['label', 'page_id']]


X_train_full = df_train_full[feature_cols]
y_train_full = df_train_full[target_col]
groups_train_full = df_train_full[group_col]

X_test = df_test[feature_cols]
# Test target might not always be present or needed for training, but keep if available
y_test = df_test[target_col] if target_col in df_test.columns else None

# 3. Encode Target Labels
# Mapping: "question", "answer", "header", "other" -> integers
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train_full)

# 4. Group-Based Train/Validation Split
# This ensures that no page_id is shared between the training and validation sets
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X_train_full, y_train_encoded, groups=groups_train_full))

X_train, y_train = X_train_full.iloc[train_idx], y_train_encoded[train_idx]
X_val, y_val = X_train_full.iloc[val_idx], y_train_encoded[val_idx]

# 5. Define Feature Weights based on your rule
feature_weights = [
    3.0 if any(k in col for k in ['node', 'x1', 'y1', 'x2', 'y2', 'center', 'width', 'height', 'area', 'aligned', 'delta', 'prev', 'next'])
    else 1.0 for col in feature_cols
]


# 6. Optuna Hyperparameter Optimization
def objective(trial):
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': len(le.classes_),
        'boosting_type': 'gbdt',
        'class_weight': 'balanced',  # <--- CRITICAL: Dynamically fixes your class imbalance
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': 0.8,  # Equivalent to feature_fraction fixed constraint
        'feature_contrib': feature_weights,  # Passes the geometric importance penalties
        'random_state': 42,
        'verbose': -1
    }

    # Initialize and train LightGBM classifier
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )

    # Evaluate using Multi-Class Log Loss on Validation Set
    preds = model.predict_proba(X_val)
    score = log_loss(y_val, preds)
    return score


# Run optimization study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)
print(f"Best Trial Multi-LogLoss: {study.best_value}")

# 7. Train Final Model with Best Parameters
best_params = study.best_params
best_params.update({
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': len(le.classes_),
    'colsample_bytree': 0.8,
    'feature_contrib': feature_weights,
    'random_state': 42,
    'verbose': -1
})


# best_params = {'n_estimators': 407, 'learning_rate': 0.06519432771811577,
# 'num_leaves': 120, 'max_depth': 10, 'min_child_samples': 30,
# 'subsample': 0.7103472738743427, 'objective': 'multiclass',
# 'metric': 'multi_logloss', 'num_class': 4,
# 'colsample_bytree': 0.8,
# 'feature_contrib': [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'random_state': 42, 'verbose': -1}


final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
)

# 8. Visualize Feature Importance
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': final_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'].head(20)[::-1], importance_df['Importance'].head(20)[::-1], color='skyblue')
plt.xlabel('Importance Value')
plt.title('Top 20 Most Important LightGBM Features')
plt.tight_layout()
plt.show()

# 9. Generate Predictions on Test Set
test_preds_encoded = final_model.predict(X_test)
df_test['predicted_label'] = le.inverse_transform(test_preds_encoded)


######################################
# Evaluation

# 1. Generate Predictions on Test Set
# Ensure the model predicts classes on the unseen test features
y_test_preds_encoded = final_model.predict(X_test)

# Decode integers back to their original text labels ("question", "answer", etc.)
y_test_true = y_test # Assumes y_test is already the original textual label string
y_test_preds = le.inverse_transform(y_test_preds_encoded)

# 2. Print Classification Report
# This shows precision, recall, and F1-score for EACH individual class
print("="*60)
print("CLASS-LEVEL PERFORMANCE REPORT")
print("="*60)
print(classification_report(y_test_true, y_test_preds))
print("="*60)

# 3. Print Class Distribution / Imbalance Check
print("\nTRAINING SET CLASS DISTRIBUTION:")
print(df_train_full[target_col].value_counts())
print("\nTEST SET CLASS DISTRIBUTION:")
print(df_test[target_col].value_counts())

# 4. Generate and Plot Confusion Matrix
# This visualizes exactly where the 13% error rate is distributed
cm = confusion_matrix(y_test_true, y_test_preds, labels=le.classes_)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=le.classes_,
    yticklabels=le.classes_
)
plt.xlabel('Predicted Labels', fontsize=12, fontweight='bold')
plt.ylabel('True Labels', fontsize=12, fontweight='bold')
plt.title('Document Form Classifier: Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

aq = 42

importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': final_model.booster_.feature_importance(importance_type='gain')})
print(importance_df.sort_values(by='Importance', ascending=False).head(20).to_string(index=False))

X_test_corr = X_test.loc[y_test_true == y_test_preds]
X_test_err = X_test.loc[y_test_true != y_test_preds]

X_test_corr_mean = X_test_corr.mean().to_frame().T
X_test_err_mean = X_test_err.mean().to_frame().T

#############################################################################################################
# ERROR ANALYSIS

# 1. Generate final test predictions and attach them to the scaled test dataframe
y_test_preds_encoded = final_model.predict(X_test)
y_test_preds = le.inverse_transform(y_test_preds_encoded)

# Make a copy of the test dataframe to safely hold diagnostic metadata
df_analysis = df_test.copy()
df_analysis['true_label'] = y_test  # Assumes y_test holds original string labels
df_analysis['pred_label'] = y_test_preds
df_analysis['is_correct'] = (df_analysis['true_label'] == df_analysis['pred_label'])

# 2. Split into Correct vs. Misclassified Dataframes as requested
df_correct = df_analysis[df_analysis['is_correct'] == True].reset_index(drop=True)
df_error = df_analysis[df_analysis['is_correct'] == False].reset_index(drop=True)

print(f"Correctly Classified Shape: {df_correct.shape}")
print(f"Misclassified Shape: {df_error.shape}\n")

# -------------------------------------------------------------------------
# DIAGNOSTIC 1: Where do the errors structurally occur? (Node Heights)
# -------------------------------------------------------------------------
print("=" * 60)
print("1. ERROR RATE BY NODE HEIGHT")
print("=" * 60)
node_stats = df_analysis.groupby('node_height')['is_correct'].agg(['count', 'mean'])
node_stats['error_rate'] = 1 - node_stats['mean']
print(node_stats[['count', 'error_rate']].to_string())

# -------------------------------------------------------------------------
# DIAGNOSTIC 2: Geometric & Punctuation Aggregates (Correct vs Error)
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. GEOMETRIC & PUNCTUATION AGGREGATES")
print("=" * 60)
geo_cols = ['rel_y_pos', 'rel_box_width', 'rel_box_height', 'rel_box_area',
            'has_question_punct', 'ends_with_colon', 'numeric_char_ratio', 'is_following_question_trigger']

# Compute the mean values for these critical layout features across both splits
mean_correct = df_correct[geo_cols].mean()
mean_error = df_error[geo_cols].mean()

summary_geo = pd.DataFrame({
    'Metric Mean (Correct)': mean_correct,
    'Metric Mean (Error)': mean_error,
    'Absolute Difference': (mean_correct - mean_error).abs()
}).sort_values(by='Absolute Difference', ascending=False)
print(summary_geo.to_string())

# -------------------------------------------------------------------------
# DIAGNOSTIC 3: Text Embedding Variance (PCA Signal Shifts)
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. TOP 5 TEXT EMBEDDING (PCA) SHIFTS")
print("=" * 60)
pca_cols = [col for col in df_analysis.columns if 'pc_' in col]

if pca_cols:
    pca_correct = df_correct[pca_cols].mean()
    pca_error = df_error[pca_cols].mean()

    summary_pca = pd.DataFrame({
        'PCA Mean (Correct)': pca_correct,
        'PCA Mean (Error)': pca_error,
        'Absolute Shift': (pca_correct - pca_error).abs()
    }).sort_values(by='Absolute Shift', ascending=False)
    print(summary_pca.head(5).to_string())
else:
    print("No PCA columns found matching 'pc_' pattern.")


print("\n" + "="*60)
print("4. TOP 5 PCA CORRELATIONS WITH ERRORS")
print("="*60)
correlations = df_analysis[pca_cols].corrwith(df_analysis['is_correct']).abs()
print(correlations.sort_values(ascending=False).head(5).to_string())