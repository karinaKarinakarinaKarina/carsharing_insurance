
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


X_train = np.load('X_train.npy')
X_val = np.load('X_val.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')
y_test = np.load('y_test.npy')

print(f"Размерность X_train: {X_train.shape}")
print(f"Размерность X_val: {X_val.shape}")
print(f"Размерность X_test: {X_test.shape}")
print(f"\nДоля ДТП в train: {y_train.mean()*100:.2f}%")
print(f"Доля ДТП в val: {y_val.mean()*100:.2f}%")
print(f"Доля ДТП в test: {y_test.mean()*100:.2f}%")

baseline_model = LogisticRegression(
    C=1.0,                      
    max_iter=1000,              
    random_state=42,            
    class_weight='balanced',    
    solver='lbfgs'             
)
baseline_model.fit(X_train, y_train)


y_train_proba = baseline_model.predict_proba(X_train)[:, 1]
y_val_proba = baseline_model.predict_proba(X_val)[:, 1]
y_test_proba = baseline_model.predict_proba(X_test)[:, 1]

print(f"\nПример предсказанных вероятностей (первые 5 тестовых поездок):")
for i in range(5):
    print(f"  Поездка {i+1}: вероятность ДТП = {y_test_proba[i]:.4f} (факт: {y_test[i]})")


train_auc = roc_auc_score(y_train, y_train_proba)
val_auc = roc_auc_score(y_val, y_val_proba)
test_auc = roc_auc_score(y_test, y_test_proba)

train_log_loss = log_loss(y_train, y_train_proba)
val_log_loss = log_loss(y_val, y_val_proba)
test_log_loss = log_loss(y_test, y_test_proba)

train_brier = brier_score_loss(y_train, y_train_proba)
val_brier = brier_score_loss(y_val, y_val_proba)
test_brier = brier_score_loss(y_test, y_test_proba)

print("\nTRAIN:")
print(f"  AUC-ROC:     {train_auc:.4f}")
print(f"  Log Loss:    {train_log_loss:.4f}")
print(f"  Brier Score: {train_brier:.4f}")

print("\nVALIDATION:")
print(f"  AUC-ROC:     {val_auc:.4f}")
print(f"  Log Loss:    {val_log_loss:.4f}")
print(f"  Brier Score: {val_brier:.4f}")

print("\nTEST:")
print(f"  AUC-ROC:     {test_auc:.4f}")
print(f"  Log Loss:    {test_log_loss:.4f}")
print(f"  Brier Score: {test_brier:.4f}")

overfit_auc = train_auc - val_auc
print(f"\nРазница AUC (train - val): {overfit_auc:.4f}")

feature_info = joblib.load('feature_info.pkl')
numeric_features = feature_info['numeric_features']
categorical_features = feature_info['categorical_features']


preprocessor = joblib.load('preprocessor.pkl')

coefficients = baseline_model.coef_[0]


feature_importance = {}
try:
    feature_names = preprocessor.get_feature_names_out()
    
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    })
    
    coef_df['abs_coef'] = coef_df['coefficient'].abs()
    coef_df = coef_df.sort_values('abs_coef', ascending=False)
    
    print("\nСамые влиятельные признаки:")
    print(coef_df.head(10)[['feature', 'coefficient']].to_string(index=False))
    
    print("\nПризнаки, увеличивающие вероятность дтп:")
    positive = coef_df[coef_df['coefficient'] > 0].head(5)
    print(positive[['feature', 'coefficient']].to_string(index=False))
    
    print("\nПризнаки, снижающие вероятность дтп:")
    negative = coef_df[coef_df['coefficient'] < 0].tail(5).sort_values('coefficient')
    print(negative[['feature', 'coefficient']].to_string(index=False))
    
except Exception as e:
    print(f"Не удалось получить названия признаков: {e}")
    
    print("\nПризнаки, увеличивающие риск:")
    numeric_coef = {}
    for i, feat in enumerate(numeric_features[:10]): 
        if i < len(coefficients):
            numeric_coef[feat] = coefficients[i]
    
    sorted_coef = sorted(numeric_coef.items(), key=lambda x: x[1], reverse=True)
    for feat, coef in sorted_coef[:5]:
        print(f"  {feat}: {coef:.4f}")
    
    print("\nПризнаки, снижающие риск:")
    for feat, coef in sorted_coef[-5:]:
        print(f"  {feat}: {coef:.4f}")


plt.figure(figsize=(10, 6))

prob_true, prob_pred = calibration_curve(y_val, y_val_proba, n_bins=10)

plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Logistic Regression')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')

plt.xlabel('Предсказанная вероятность ДТП')
plt.ylabel('Фактическая доля ДТП')
plt.title('Калибровочная кривая (Validation)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('baseline_calibration_curve.png', dpi=150)
print("\nАнализ калибровки:")
for i, (pred, true) in enumerate(zip(prob_pred, prob_true)):
    print(f"  Бин {i+1}: предсказано {pred:.3f}, фактически {true:.3f}, разница = {abs(pred - true):.3f}")


joblib.dump(baseline_model, 'baseline_model.pkl')

metrics_dict = {
    'train': {
        'auc': train_auc,
        'log_loss': train_log_loss,
        'brier_score': train_brier
    },
    'val': {
        'auc': val_auc,
        'log_loss': val_log_loss,
        'brier_score': val_brier
    },
    'test': {
        'auc': test_auc,
        'log_loss': test_log_loss,
        'brier_score': test_brier
    }
}

joblib.dump(metrics_dict, 'baseline_metrics.pkl')

print(f"""
Baseline модель (Logistic Regression) для прогнозирования вероятности ДТП:

Финальные метрики:
  AUC-ROC:     {test_auc:.4f}
  Log Loss:    {test_log_loss:.4f}
  Brier Score: {test_brier:.4f}
""")