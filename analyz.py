import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc
from sklearn.calibration import calibration_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

preprocessor = joblib.load('models/preprocessor.pkl')
best_model_info = joblib.load('models/best_model_info.pkl')
model = joblib.load('models/best_model.pkl')

print(f"\nМодель: {best_model_info['model_name']}")
print(f"Test AUC: {best_model_info['test_auc']:.4f}")
print(f"Test Log Loss: {best_model_info['test_log_loss']:.4f}")
print(f"Test Brier Score: {best_model_info['test_brier_score']:.4f}")

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)


cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")


precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn)
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision:    {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:       {recall:.4f} ({recall*100:.2f}%)")
print(f"Specificity:  {specificity:.4f} ({specificity*100:.2f}%)")
print(f"F1-Score:     {f1:.4f}")

total_errors = fp + fn
print(f"Всего ошибок: {total_errors} из {len(y_test)} ({total_errors/len(y_test)*100:.2f}%)")
print(f"Ложные срабатывания (FP): {fp} ({fp/len(y_test)*100:.2f}%)")
print(f"Пропущенные ДТП (FN): {fn} ({fn/len(y_test)*100:.2f}%)")
print(f"Доля пропущенных ДТП среди всех ДТП: {fn/y_test.sum()*100:.2f}%")


thresholds = np.arange(0.1, 0.9, 0.05)
threshold_results = []

for thresh in thresholds:
    y_pred_thresh = (y_pred_proba >= thresh).astype(int)
    fp_thresh = ((y_pred_thresh == 1) & (y_test == 0)).sum()
    fn_thresh = ((y_pred_thresh == 0) & (y_test == 1)).sum()
    prec_thresh = (y_pred_thresh.sum() > 0) and ((y_pred_thresh & y_test).sum() / y_pred_thresh.sum()) or 0
    rec_thresh = (y_pred_thresh & y_test).sum() / y_test.sum() if y_test.sum() > 0 else 0
    f1_thresh = 2 * (prec_thresh * rec_thresh) / (prec_thresh + rec_thresh) if (prec_thresh + rec_thresh) > 0 else 0
    
    threshold_results.append({
        'threshold': thresh,
        'FP': fp_thresh,
        'FN': fn_thresh,
        'precision': prec_thresh,
        'recall': rec_thresh,
        'f1': f1_thresh
    })
    
    print(f"Порог {thresh:.2f}: FP={fp_thresh}, FN={fn_thresh}, Precision={prec_thresh:.3f}, Recall={rec_thresh:.3f}, F1={f1_thresh:.3f}")

threshold_df = pd.DataFrame(threshold_results)
best_f1_idx = threshold_df['f1'].idxmax()
best_threshold = threshold_df.loc[best_f1_idx, 'threshold']
print(f"\nОптимальный порог по F1-score: {best_threshold:.2f}")

fig = plt.figure(figsize=(20, 12))

ax1 = plt.subplot(2, 4, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['No Accident', 'Accident'],
            yticklabels=['No Accident', 'Accident'])
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')
ax1.set_title('Confusion Matrix')

ax2 = plt.subplot(2, 4, 2)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
ax2.plot(fpr, tpr, linewidth=2, color='darkorange', label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(2, 4, 3)
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall_vals, precision_vals)
ax3.plot(recall_vals, precision_vals, linewidth=2, color='green', label=f'PR Curve (AUC = {pr_auc:.4f})')
ax3.set_xlabel('Recall')
ax3.set_ylabel('Precision')
ax3.set_title('Precision-Recall Curve')
ax3.legend(loc='lower left')
ax3.grid(True, alpha=0.3)

ax4 = plt.subplot(2, 4, 4)
prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
ax4.plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=8, label='Model')
ax4.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
ax4.set_xlabel('Predicted Probability')
ax4.set_ylabel('Actual Fraction')
ax4.set_title('Calibration Curve')
ax4.legend()
ax4.grid(True, alpha=0.3)

ax5 = plt.subplot(2, 4, 5)
error_indices = np.where(y_pred != y_test)[0]
error_probs = y_pred_proba[error_indices]
error_true = y_test[error_indices]
error_type = ['False Positive' if y_true == 0 else 'False Negative' for y_true in error_true]

error_df = pd.DataFrame({
    'index': error_indices,
    'probability': error_probs,
    'true_label': error_true,
    'error_type': error_type
})

fp_df = error_df[error_df['error_type'] == 'False Positive']
fn_df = error_df[error_df['error_type'] == 'False Negative']

ax5.hist(fp_df['probability'], bins=20, alpha=0.7, label=f'False Positives (n={len(fp_df)})', color='red', edgecolor='black')
ax5.hist(fn_df['probability'], bins=20, alpha=0.7, label=f'False Negatives (n={len(fn_df)})', color='blue', edgecolor='black')
ax5.set_xlabel('Predicted Probability')
ax5.set_ylabel('Count')
ax5.set_title('Error Distribution by Probability')
ax5.legend()
ax5.grid(True, alpha=0.3)

ax6 = plt.subplot(2, 4, 6)
error_by_prob = []
for thresh in np.arange(0, 1.1, 0.1):
    fp_in_range = ((fp_df['probability'] >= thresh) & (fp_df['probability'] < thresh + 0.1)).sum()
    fn_in_range = ((fn_df['probability'] >= thresh) & (fn_df['probability'] < thresh + 0.1)).sum()
    error_by_prob.append({'range': f'{thresh:.1f}-{thresh+0.1:.1f}', 'FP': fp_in_range, 'FN': fn_in_range})

error_by_prob_df = pd.DataFrame(error_by_prob)
x = np.arange(len(error_by_prob_df))
width = 0.35
ax6.bar(x - width/2, error_by_prob_df['FP'], width, label='False Positives', color='red', edgecolor='black')
ax6.bar(x + width/2, error_by_prob_df['FN'], width, label='False Negatives', color='blue', edgecolor='black')
ax6.set_xlabel('Probability Range')
ax6.set_ylabel('Number of Errors')
ax6.set_title('Errors by Probability Range')
ax6.set_xticks(x)
ax6.set_xticklabels(error_by_prob_df['range'], rotation=45)
ax6.legend()
ax6.grid(True, alpha=0.3)

ax7 = plt.subplot(2, 4, 7)
threshold_df.plot(x='threshold', y=['precision', 'recall', 'f1'], ax=ax7, marker='o', linewidth=2)
ax7.axvline(x=best_threshold, color='red', linestyle='--', alpha=0.7, label=f'Best F1 threshold = {best_threshold:.2f}')
ax7.set_xlabel('Threshold')
ax7.set_ylabel('Score')
ax7.set_title('Precision, Recall, F1 vs Threshold')
ax7.legend()
ax7.grid(True, alpha=0.3)

ax8 = plt.subplot(2, 4, 8)
error_rates = []
for thresh in thresholds:
    y_pred_thresh = (y_pred_proba >= thresh).astype(int)
    fp_rate = ((y_pred_thresh == 1) & (y_test == 0)).sum() / (y_test == 0).sum()
    fn_rate = ((y_pred_thresh == 0) & (y_test == 1)).sum() / (y_test == 1).sum()
    error_rates.append({'threshold': thresh, 'FPR': fp_rate, 'FNR': fn_rate})

error_rates_df = pd.DataFrame(error_rates)
ax8.plot(error_rates_df['threshold'], error_rates_df['FPR'], marker='o', linewidth=2, label='False Positive Rate', color='red')
ax8.plot(error_rates_df['threshold'], error_rates_df['FNR'], marker='s', linewidth=2, label='False Negative Rate', color='blue')
ax8.axvline(x=best_threshold, color='green', linestyle='--', alpha=0.7, label=f'Best threshold = {best_threshold:.2f}')
ax8.set_xlabel('Threshold')
ax8.set_ylabel('Error Rate')
ax8.set_title('Error Rates vs Threshold')
ax8.legend()
ax8.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('error_analysis_full.png', dpi=150)

plt.figure(figsize=(12, 8))
metrics_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
metrics_values = [accuracy, precision, recall, specificity, f1]
colors = ['steelblue', 'coral', 'seagreen', 'goldenrod', 'purple']

bars = plt.bar(metrics_names, metrics_values, color=colors, edgecolor='black')
plt.ylim(0, 1.1)
plt.ylabel('Score')
plt.title('Model Performance Metrics')
for bar, val in zip(bars, metrics_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', va='bottom')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('performance_metrics.png', dpi=150)
plt.close()


test_df = pd.read_csv('car_sharing_insurance_complete.csv')
test_df['trip_start_datetime'] = pd.to_datetime(test_df['trip_start_datetime'])
test_df = test_df.sort_values('trip_start_datetime').reset_index(drop=True)
split_date = test_df['trip_start_datetime'].quantile(0.8)
test_df = test_df[test_df['trip_start_datetime'] > split_date].reset_index(drop=True)

test_df = test_df.iloc[:len(y_test)].copy()
test_df['predicted_proba'] = y_pred_proba
test_df['predicted_class'] = y_pred
test_df['error'] = (y_pred != y_test).astype(int)
test_df['error_type'] = error_type + ['Correct'] * (len(test_df) - len(error_type))


fp_samples = test_df[test_df['error_type'] == 'False Positive'].head(10)
fn_samples = test_df[test_df['error_type'] == 'False Negative'].head(10)

print("\nЛожные срабатывания (модель предсказала ДТП, но его не было):")
if len(fp_samples) > 0:
    cols_to_show = ['trip_start_datetime', 'driver_age', 'driving_style', 'weather_condition', 'is_night_drive', 'predicted_proba']
    print(fp_samples[cols_to_show].to_string(index=False))
else:
    print("Нет ложных срабатываний")

print("\nПропущенные ДТП (модель не предсказала ДТП, но оно произошло):")
if len(fn_samples) > 0:
    cols_to_show = ['trip_start_datetime', 'driver_age', 'driving_style', 'weather_condition', 'is_night_drive', 'predicted_proba']
    print(fn_samples[cols_to_show].to_string(index=False))
else:
    print("Нет пропущенных ДТП")


test_df['age_group'] = pd.cut(test_df['driver_age'], bins=[18, 25, 35, 50, 100], labels=['18-25', '26-35', '36-50', '50+'])
fp_by_age = test_df[test_df['error_type'] == 'False Positive'].groupby('age_group').size()
fn_by_age = test_df[test_df['error_type'] == 'False Negative'].groupby('age_group').size()
total_by_age = test_df.groupby('age_group').size()

print("\nОшибки по возрастным группам:")
for age in total_by_age.index:
    fp_rate = fp_by_age.get(age, 0) / total_by_age[age] * 100
    fn_rate = fn_by_age.get(age, 0) / total_by_age[age] * 100
    print(f"  {age}: FP={fp_by_age.get(age, 0)} ({fp_rate:.1f}%), FN={fn_by_age.get(age, 0)} ({fn_rate:.1f}%)")

fp_by_style = test_df[test_df['error_type'] == 'False Positive'].groupby('driving_style').size()
fn_by_style = test_df[test_df['error_type'] == 'False Negative'].groupby('driving_style').size()
total_by_style = test_df.groupby('driving_style').size()

print("\nОшибки по стилю вождения:")
for style in total_by_style.index:
    fp_rate = fp_by_style.get(style, 0) / total_by_style[style] * 100
    fn_rate = fn_by_style.get(style, 0) / total_by_style[style] * 100
    print(f"  {style}: FP={fp_by_style.get(style, 0)} ({fp_rate:.1f}%), FN={fn_by_style.get(style, 0)} ({fn_rate:.1f}%)")

fp_by_weather = test_df[test_df['error_type'] == 'False Positive'].groupby('weather_condition').size()
fn_by_weather = test_df[test_df['error_type'] == 'False Negative'].groupby('weather_condition').size()
total_by_weather = test_df.groupby('weather_condition').size()

print("\nОшибки по погодным условиям:")
for weather in total_by_weather.index:
    fp_rate = fp_by_weather.get(weather, 0) / total_by_weather[weather] * 100
    fn_rate = fn_by_weather.get(weather, 0) / total_by_weather[weather] * 100
    print(f"  {weather}: FP={fp_by_weather.get(weather, 0)} ({fp_rate:.1f}%), FN={fn_by_weather.get(weather, 0)} ({fn_rate:.1f}%)")

fp_by_night = test_df[test_df['error_type'] == 'False Positive'].groupby('is_night_drive').size()
fn_by_night = test_df[test_df['error_type'] == 'False Negative'].groupby('is_night_drive').size()
total_by_night = test_df.groupby('is_night_drive').size()

print("\nОшибки по времени суток:")
for night in total_by_night.index:
    night_label = 'Night' if night == 1 else 'Day'
    fp_rate = fp_by_night.get(night, 0) / total_by_night[night] * 100
    fn_rate = fn_by_night.get(night, 0) / total_by_night[night] * 100
    print(f"  {night_label}: FP={fp_by_night.get(night, 0)} ({fp_rate:.1f}%), FN={fn_by_night.get(night, 0)} ({fn_rate:.1f}%)")


print(f"Средняя предсказанная вероятность для ДТП: {y_pred_proba[y_test==1].mean():.4f}")
print(f"Средняя предсказанная вероятность для не-ДТП: {y_pred_proba[y_test==0].mean():.4f}")
print(f"Медианная вероятность для ДТП: {np.median(y_pred_proba[y_test==1]):.4f}")
print(f"Медианная вероятность для не-ДТП: {np.median(y_pred_proba[y_test==0]):.4f}")
print(f"Минимальная вероятность среди ДТП: {y_pred_proba[y_test==1].min():.4f}")
print(f"Максимальная вероятность среди не-ДТП: {y_pred_proba[y_test==0].max():.4f}")

