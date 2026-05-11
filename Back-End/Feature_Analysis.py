import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("Feature Importance Analysis")
print("=" * 70)

# Load models
with open('models/random_forest.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('models/xgboost.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# Feature names
feature_names = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
                'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
                'root_shell','su_attempted','num_root','num_file_creations','num_shells',
                'num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count',
                'srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
                'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
                'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
                'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate']

# Get feature importance
rf_importance = rf_model.feature_importances_
xgb_importance = xgb_model.feature_importances_

# Average importance
avg_importance = (rf_importance + xgb_importance) / 2

# Create DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': avg_importance
}).sort_values('importance', ascending=False)

print("\n Top 15 Most Important Features:")
print("=" * 70)
for idx, row in importance_df.head(15).iterrows():
    print(f"{row['feature']:30s} : {row['importance']:.4f}")

# Save
importance_df.to_csv('models/feature_importance.csv', index=False)
print("\n Saved: models/feature_importance.csv")

# Plot
plt.figure(figsize=(12, 8))
sns.barplot(data=importance_df.head(15), x='importance', y='feature', palette='viridis')
plt.title('Top 15 Feature Importance', fontsize=16, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
print(" Saved: models/feature_importance.png")

print("\n" + "=" * 70)
print(" Analysis Complete!")
print("=" * 70)