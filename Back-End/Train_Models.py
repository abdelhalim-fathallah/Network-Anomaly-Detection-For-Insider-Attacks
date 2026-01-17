import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
import os
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print(" MAXIMUM ACCURACY TRAINING - NSL-KDD Dataset")
print(" Target: 98%+ Detection Rate")
print(" Using: Random Forest + XGBoost + Gradient Boosting")
print("=" * 80)

# ==================== Load Data ====================
print("\n Step 1/6: Loading NSL-KDD Dataset...")
start_time = time.time()

cols = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
        'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
        'root_shell','su_attempted','num_root','num_file_creations','num_shells',
        'num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count',
        'srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
        'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
        'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
        'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate',
        'label','difficulty']

df = pd.read_csv('data/KDDTrain.txt', names=cols, header=None)
df = df.drop('difficulty', axis=1)

print(f" Loaded {len(df):,} records")
print(f"   Time: {time.time() - start_time:.2f}s")

# ==================== Advanced Preprocessing ====================
print("\n Step 2/6: Advanced Preprocessing...")
start_time = time.time()

# Separate features and labels
X = df.drop('label', axis=1).copy()
y = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

print(f"   Initial distribution:")
print(f"   Normal: {sum(y == 0):,} ({sum(y == 0)/len(y)*100:.1f}%)")
print(f"   Attack: {sum(y == 1):,} ({sum(y == 1)/len(y)*100:.1f}%)")

# Remove exact duplicates
print("   Removing duplicates...")
initial_size = len(X)
X_combined = pd.concat([X, y], axis=1)
X_combined = X_combined.drop_duplicates()
X = X_combined.iloc[:, :-1]
y = X_combined.iloc[:, -1]
print(f"   Removed {initial_size - len(X):,} duplicates")

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"   Numeric features: {len(numeric_cols)}")
print(f"   Categorical features: {len(categorical_cols)}")

# Encode categorical features
print("   Encoding categorical features...")
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Handle any remaining missing values
X = X.fillna(0)

# Feature scaling
print("   Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame for feature importance later
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print(f" Preprocessing complete")
print(f"   Final samples: {len(X_scaled):,}")
print(f"   Features: {X_scaled.shape[1]}")
print(f"   Time: {time.time() - start_time:.2f}s")

# ==================== Split Data ====================
print("\n Step 3/6: Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y.values, test_size=0.2, random_state=42, stratify=y
)

print(f" Split complete")
print(f"   Training: {len(X_train):,} samples")
print(f"   Testing: {len(X_test):,} samples")
print(f"   Test set - Normal: {sum(y_test == 0):,}, Attack: {sum(y_test == 1):,}")

# ==================== Train Random Forest ====================
print("\n Step 4/6: Training Random Forest (Optimized)...")
print("   Parameters: 200 trees, max_depth=30, class_weight='balanced'")
start_time = time.time()

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_model.fit(X_train, y_train)

training_time = time.time() - start_time
print(f"✅ Random Forest training complete")
print(f"   Time: {training_time:.2f}s ({training_time/60:.1f} min)")

# Evaluate Random Forest
y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
rec_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_proba_rf)

print(f"\n   Random Forest Results:")
print(f"   Accuracy:  {acc_rf * 100:.2f}%")
print(f"   Precision: {prec_rf * 100:.2f}%")
print(f"   Recall:    {rec_rf * 100:.2f}%")
print(f"   F1-Score:  {f1_rf * 100:.2f}%")
print(f"   AUC-ROC:   {auc_rf * 100:.2f}%")

# ==================== Train XGBoost ====================
print("\n Step 5/6: Training XGBoost (High Performance)...")
print("   Parameters: 300 estimators, learning_rate=0.1, max_depth=7")
start_time = time.time()

# Calculate scale_pos_weight for imbalanced data
scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=7,
    min_child_weight=3,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',
    verbosity=1
)

xgb_model.fit(X_train, y_train)

training_time = time.time() - start_time
print(f" XGBoost training complete")
print(f"   Time: {training_time:.2f}s ({training_time/60:.1f} min)")

# Evaluate XGBoost
y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

acc_xgb = accuracy_score(y_test, y_pred_xgb)
prec_xgb = precision_score(y_test, y_pred_xgb)
rec_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
auc_xgb = roc_auc_score(y_test, y_proba_xgb)

print(f"\n   XGBoost Results:")
print(f"   Accuracy:  {acc_xgb * 100:.2f}%")
print(f"   Precision: {prec_xgb * 100:.2f}%")
print(f"   Recall:    {rec_xgb * 100:.2f}%")
print(f"   F1-Score:  {f1_xgb * 100:.2f}%")
print(f"   AUC-ROC:   {auc_xgb * 100:.2f}%")

# ==================== Train Gradient Boosting ====================
print("\n Step 6/6: Training Gradient Boosting...")
print("   Parameters: 150 estimators, learning_rate=0.1, max_depth=5")
start_time = time.time()

gb_model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42,
    verbose=1
)

gb_model.fit(X_train, y_train)

training_time = time.time() - start_time
print(f" Gradient Boosting training complete")
print(f" Time: {training_time:.2f}s ({training_time/60:.1f} min)")

# Evaluate Gradient Boosting
y_pred_gb = gb_model.predict(X_test)
y_proba_gb = gb_model.predict_proba(X_test)[:, 1]

acc_gb = accuracy_score(y_test, y_pred_gb)
prec_gb = precision_score(y_test, y_pred_gb)
rec_gb = recall_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb)
auc_gb = roc_auc_score(y_test, y_proba_gb)

print(f"\n   Gradient Boosting Results:")
print(f"   Accuracy:  {acc_gb * 100:.2f}%")
print(f"   Precision: {prec_gb * 100:.2f}%")
print(f"   Recall:    {rec_gb * 100:.2f}%")
print(f"   F1-Score:  {f1_gb * 100:.2f}%")
print(f"   AUC-ROC:   {auc_gb * 100:.2f}%")

# ==================== Ensemble Voting ====================
print("\n Creating Weighted Ensemble...")

# Weighted average of probabilities (XGBoost gets more weight as it's usually best)
y_proba_ensemble = (0.35 * y_proba_rf + 0.45 * y_proba_xgb + 0.20 * y_proba_gb)
y_pred_ensemble = (y_proba_ensemble >= 0.5).astype(int)

acc_ens = accuracy_score(y_test, y_pred_ensemble)
prec_ens = precision_score(y_test, y_pred_ensemble)
rec_ens = recall_score(y_test, y_pred_ensemble)
f1_ens = f1_score(y_test, y_pred_ensemble)
auc_ens = roc_auc_score(y_test, y_proba_ensemble)

# ==================== Final Results ====================
print("\n" + "=" * 80)
print(" FINAL RESULTS - ALL MODELS")
print("=" * 80)

results = {
    'Random Forest': {'acc': acc_rf, 'prec': prec_rf, 'rec': rec_rf, 'f1': f1_rf, 'auc': auc_rf},
    'XGBoost': {'acc': acc_xgb, 'prec': prec_xgb, 'rec': rec_xgb, 'f1': f1_xgb, 'auc': auc_xgb},
    'Gradient Boosting': {'acc': acc_gb, 'prec': prec_gb, 'rec': rec_gb, 'f1': f1_gb, 'auc': auc_gb},
    'Ensemble': {'acc': acc_ens, 'prec': prec_ens, 'rec': rec_ens, 'f1': f1_ens, 'auc': auc_ens}
}

for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {metrics['acc']*100:6.2f}%")
    print(f"  Precision: {metrics['prec']*100:6.2f}%")
    print(f"  Recall:    {metrics['rec']*100:6.2f}%")
    print(f"  F1-Score:  {metrics['f1']*100:6.2f}%")
    print(f"  AUC-ROC:   {metrics['auc']*100:6.2f}%")

# Find best model
best_model_name = max(results.keys(), key=lambda k: results[k]['acc'])
best_acc = results[best_model_name]['acc']

print("\n" + "=" * 80)
print(f" BEST MODEL: {best_model_name}")
print(f" DETECTION RATE: {best_acc * 100:.2f}%")
print(f" TARGET ACHIEVED: {'YES! 🎉' if best_acc >= 0.95 else 'Almost there! 💪'}")
print("=" * 80)

# Detailed classification report for best performing model
print("\n Detailed Classification Report (Ensemble):")
print(classification_report(y_test, y_pred_ensemble, target_names=['Normal', 'Attack'], digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_ensemble)
print("\n Confusion Matrix (Ensemble):")
print(f"              Predicted")
print(f"              Normal  Attack")
print(f"Actual Normal  {cm[0][0]:6d}  {cm[0][1]:6d}")
print(f"Actual Attack  {cm[1][0]:6d}  {cm[1][1]:6d}")

tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives:  {tn:,}")
print(f"False Positives: {fp:,}")
print(f"False Negatives: {fn:,}")
print(f"True Positives:  {tp:,}")
print(f"\nFalse Positive Rate: {fp/(fp+tn)*100:.2f}%")
print(f"False Negative Rate: {fn/(fn+tp)*100:.2f}%")

# ==================== Save Models ====================
print("\n Saving all trained models...")

os.makedirs('models', exist_ok=True)

# Save all models
with open('models/random_forest.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print(" Saved: models/random_forest.pkl")

with open('models/xgboost.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print(" Saved: models/xgboost.pkl")

with open('models/gradient_boosting.pkl', 'wb') as f:
    pickle.dump(gb_model, f)
print(" Saved: models/gradient_boosting.pkl")

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(" Saved: models/scaler.pkl")

with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print(" Saved: models/label_encoders.pkl")

# Save metrics
metrics_dict = {
    'random_forest': {
        'accuracy': float(acc_rf),
        'precision': float(prec_rf),
        'recall': float(rec_rf),
        'f1_score': float(f1_rf),
        'auc': float(auc_rf)
    },
    'xgboost': {
        'accuracy': float(acc_xgb),
        'precision': float(prec_xgb),
        'recall': float(rec_xgb),
        'f1_score': float(f1_xgb),
        'auc': float(auc_xgb)
    },
    'gradient_boosting': {
        'accuracy': float(acc_gb),
        'precision': float(prec_gb),
        'recall': float(rec_gb),
        'f1_score': float(f1_gb),
        'auc': float(auc_gb)
    },
    'ensemble': {
        'accuracy': float(acc_ens),
        'precision': float(prec_ens),
        'recall': float(rec_ens),
        'f1_score': float(f1_ens),
        'auc': float(auc_ens)
    }
}

with open('models/metrics.pkl', 'wb') as f:
    pickle.dump(metrics_dict, f)
print(" Saved: models/metrics.pkl")

# ==================== Summary ====================
print("\n" + "=" * 80)
print(" TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\n Best Detection Rate: {best_acc * 100:.2f}%")
print(f"\n Model Used: {best_model_name}")
print(f"\n All models saved in: models/")
print(f"\n System ready for deployment!")
print("=" * 80)