import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.catboost
import mlflow.lightgbm
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("car_sharing_insurance_experiments")

X_train = np.load('X_train.npy')
X_val = np.load('X_val.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')
y_test = np.load('y_test.npy')

def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, model_name="model"):
    if hasattr(model, "predict_proba"):
        y_train_pred = model.predict_proba(X_train)[:, 1]
        y_val_pred = model.predict_proba(X_val)[:, 1]
        y_test_pred = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "predict"):
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
    else:
        raise ValueError("Модель не поддерживает predict_proba или predict")
    
    metrics = {
        'train': {
            'auc': roc_auc_score(y_train, y_train_pred),
            'log_loss': log_loss(y_train, y_train_pred),
            'brier': brier_score_loss(y_train, y_train_pred)
        },
        'val': {
            'auc': roc_auc_score(y_val, y_val_pred),
            'log_loss': log_loss(y_val, y_val_pred),
            'brier': brier_score_loss(y_val, y_val_pred)
        },
        'test': {
            'auc': roc_auc_score(y_test, y_test_pred),
            'log_loss': log_loss(y_test, y_test_pred),
            'brier': brier_score_loss(y_test, y_test_pred)
        }
    }
    
    return metrics, y_val_pred, y_test_pred

def measure_latency(model, X_test, n_iterations=100):
    latencies = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        if hasattr(model, "predict_proba"):
            model.predict_proba(X_test)
        else:
            model.predict(X_test)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    return np.mean(latencies), np.std(latencies), np.percentile(latencies, 95)

def log_experiment(model, model_name, params, X_train, X_val, X_test, y_train, y_val, y_test, latency_mean, latency_std, latency_p95):
    with mlflow.start_run(run_name=f"{model_name}"):
        mlflow.log_params(params)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_metrics({
            "latency_mean_ms": latency_mean,
            "latency_std_ms": latency_std,
            "latency_p95_ms": latency_p95
        })
        
        metrics, y_val_pred, y_test_pred = evaluate_model(
            model, X_train, X_val, X_test, y_train, y_val, y_test, model_name
        )
        
        mlflow.log_metrics({
            "train_auc": metrics['train']['auc'],
            "train_log_loss": metrics['train']['log_loss'],
            "train_brier": metrics['train']['brier'],
            "val_auc": metrics['val']['auc'],
            "val_log_loss": metrics['val']['log_loss'],
            "val_brier": metrics['val']['brier'],
            "test_auc": metrics['test']['auc'],
            "test_log_loss": metrics['test']['log_loss'],
            "test_brier": metrics['test']['brier']
        })
        
        overfit_auc = metrics['train']['auc'] - metrics['val']['auc']
        mlflow.log_metric("overfit_auc", overfit_auc)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        prob_true, prob_pred = calibration_curve(y_val, y_val_pred, n_bins=10)
        ax.plot(prob_pred, prob_true, marker='o', linewidth=2, label=model_name)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Actual Fraction')
        ax.set_title(f'Calibration Curve - {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'calibration_curve_{model_name}.png', dpi=150)
        mlflow.log_artifact(f'calibration_curve_{model_name}.png')
        plt.close()
        
        mlflow.sklearn.log_model(
            model, 
            model_name,
            input_example=X_train[:5],
            signature=mlflow.models.infer_signature(
                X_train[:5], 
                model.predict_proba(X_train[:5]) if hasattr(model, 'predict_proba') else model.predict(X_train[:5])
            )
        )
        
        metrics_df = pd.DataFrame({
            'dataset': ['train', 'val', 'test'],
            'auc': [metrics['train']['auc'], metrics['val']['auc'], metrics['test']['auc']],
            'log_loss': [metrics['train']['log_loss'], metrics['val']['log_loss'], metrics['test']['log_loss']],
            'brier_score': [metrics['train']['brier'], metrics['val']['brier'], metrics['test']['brier']]
        })
        metrics_df.to_csv(f'metrics_{model_name}.csv', index=False)
        mlflow.log_artifact(f'metrics_{model_name}.csv')
        
        return metrics

if not os.path.exists('models'):
    os.makedirs('models')

params_lr = {
    'C': 1.0,
    'max_iter': 1000,
    'random_state': 42,
    'class_weight': 'balanced',
    'solver': 'lbfgs'
}

model_lr = LogisticRegression(**params_lr)
model_lr.fit(X_train, y_train)
latency_mean, latency_std, latency_p95 = measure_latency(model_lr, X_test)
metrics_lr = log_experiment(model_lr, "LogisticRegression", params_lr, X_train, X_val, X_test, y_train, y_val, y_test, latency_mean, latency_std, latency_p95)

base_lr = LogisticRegression(**params_lr)
model_lr_cal = CalibratedClassifierCV(base_lr, method='sigmoid', cv=3)
model_lr_cal.fit(X_train, y_train)
latency_mean, latency_std, latency_p95 = measure_latency(model_lr_cal, X_test)
metrics_lr_cal = log_experiment(model_lr_cal, "LogisticRegression_Calibrated", params_lr, X_train, X_val, X_test, y_train, y_val, y_test, latency_mean, latency_std, latency_p95)

params_rf = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 50,
    'min_samples_leaf': 20,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

model_rf = RandomForestClassifier(**params_rf)
model_rf.fit(X_train, y_train)
latency_mean, latency_std, latency_p95 = measure_latency(model_rf, X_test)
metrics_rf = log_experiment(model_rf, "RandomForest", params_rf, X_train, X_val, X_test, y_train, y_val, y_test, latency_mean, latency_std, latency_p95)

params_rf_opt = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 30,
    'min_samples_leaf': 10,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

model_rf_opt = RandomForestClassifier(**params_rf_opt)
model_rf_opt.fit(X_train, y_train)
latency_mean, latency_std, latency_p95 = measure_latency(model_rf_opt, X_test)
metrics_rf_opt = log_experiment(model_rf_opt, "RandomForest_Optimized", params_rf_opt, X_train, X_val, X_test, y_train, y_val, y_test, latency_mean, latency_std, latency_p95)

params_xgb = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': (1 - y_train.mean()) / y_train.mean(),
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

model_xgb = xgb.XGBClassifier(**params_xgb)
model_xgb.fit(X_train, y_train)
latency_mean, latency_std, latency_p95 = measure_latency(model_xgb, X_test)
metrics_xgb = log_experiment(model_xgb, "XGBoost", params_xgb, X_train, X_val, X_test, y_train, y_val, y_test, latency_mean, latency_std, latency_p95)

params_xgb_opt = {
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'scale_pos_weight': (1 - y_train.mean()) / y_train.mean(),
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

model_xgb_opt = xgb.XGBClassifier(**params_xgb_opt)
model_xgb_opt.fit(X_train, y_train)
latency_mean, latency_std, latency_p95 = measure_latency(model_xgb_opt, X_test)
metrics_xgb_opt = log_experiment(model_xgb_opt, "XGBoost_Optimized", params_xgb_opt, X_train, X_val, X_test, y_train, y_val, y_test, latency_mean, latency_std, latency_p95)

params_cat = {
    'iterations': 100,
    'depth': 6,
    'learning_rate': 0.1,
    'loss_function': 'Logloss',
    'auto_class_weights': 'Balanced',
    'random_seed': 42,
    'verbose': False
}

model_cat = CatBoostClassifier(**params_cat)
model_cat.fit(X_train, y_train)
latency_mean, latency_std, latency_p95 = measure_latency(model_cat, X_test)
metrics_cat = log_experiment(model_cat, "CatBoost", params_cat, X_train, X_val, X_test, y_train, y_val, y_test, latency_mean, latency_std, latency_p95)

params_lgb = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'class_weight': 'balanced',
    'random_state': 42,
    'verbose': -1
}

model_lgb = lgb.LGBMClassifier(**params_lgb)
model_lgb.fit(X_train, y_train)
latency_mean, latency_std, latency_p95 = measure_latency(model_lgb, X_test)
metrics_lgb = log_experiment(model_lgb, "LightGBM", params_lgb, X_train, X_val, X_test, y_train, y_val, y_test, latency_mean, latency_std, latency_p95)

all_results = []

all_results.append({
    'model': 'LogisticRegression',
    'auc': metrics_lr['test']['auc'],
    'log_loss': metrics_lr['test']['log_loss'],
    'brier_score': metrics_lr['test']['brier'],
    'latency_mean_ms': measure_latency(model_lr, X_test)[0],
    'latency_p95_ms': measure_latency(model_lr, X_test)[2]
})

all_results.append({
    'model': 'LogisticRegression_Calibrated',
    'auc': metrics_lr_cal['test']['auc'],
    'log_loss': metrics_lr_cal['test']['log_loss'],
    'brier_score': metrics_lr_cal['test']['brier'],
    'latency_mean_ms': measure_latency(model_lr_cal, X_test)[0],
    'latency_p95_ms': measure_latency(model_lr_cal, X_test)[2]
})

all_results.append({
    'model': 'RandomForest',
    'auc': metrics_rf['test']['auc'],
    'log_loss': metrics_rf['test']['log_loss'],
    'brier_score': metrics_rf['test']['brier'],
    'latency_mean_ms': measure_latency(model_rf, X_test)[0],
    'latency_p95_ms': measure_latency(model_rf, X_test)[2]
})

all_results.append({
    'model': 'RandomForest_Optimized',
    'auc': metrics_rf_opt['test']['auc'],
    'log_loss': metrics_rf_opt['test']['log_loss'],
    'brier_score': metrics_rf_opt['test']['brier'],
    'latency_mean_ms': measure_latency(model_rf_opt, X_test)[0],
    'latency_p95_ms': measure_latency(model_rf_opt, X_test)[2]
})

all_results.append({
    'model': 'XGBoost',
    'auc': metrics_xgb['test']['auc'],
    'log_loss': metrics_xgb['test']['log_loss'],
    'brier_score': metrics_xgb['test']['brier'],
    'latency_mean_ms': measure_latency(model_xgb, X_test)[0],
    'latency_p95_ms': measure_latency(model_xgb, X_test)[2]
})

all_results.append({
    'model': 'XGBoost_Optimized',
    'auc': metrics_xgb_opt['test']['auc'],
    'log_loss': metrics_xgb_opt['test']['log_loss'],
    'brier_score': metrics_xgb_opt['test']['brier'],
    'latency_mean_ms': measure_latency(model_xgb_opt, X_test)[0],
    'latency_p95_ms': measure_latency(model_xgb_opt, X_test)[2]
})

all_results.append({
    'model': 'CatBoost',
    'auc': metrics_cat['test']['auc'],
    'log_loss': metrics_cat['test']['log_loss'],
    'brier_score': metrics_cat['test']['brier'],
    'latency_mean_ms': measure_latency(model_cat, X_test)[0],
    'latency_p95_ms': measure_latency(model_cat, X_test)[2]
})

all_results.append({
    'model': 'LightGBM',
    'auc': metrics_lgb['test']['auc'],
    'log_loss': metrics_lgb['test']['log_loss'],
    'brier_score': metrics_lgb['test']['brier'],
    'latency_mean_ms': measure_latency(model_lgb, X_test)[0],
    'latency_p95_ms': measure_latency(model_lgb, X_test)[2]
})

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('auc', ascending=False)

results_df.to_csv('experiments_results.csv', index=False)

best_model_name = results_df.iloc[0]['model']
best_model_auc = results_df.iloc[0]['auc']
best_model_log_loss = results_df.iloc[0]['log_loss']
best_model_brier = results_df.iloc[0]['brier_score']

if best_model_name == 'LogisticRegression_Calibrated':
    joblib.dump(model_lr_cal, 'models/best_model.pkl')
elif best_model_name == 'LogisticRegression':
    joblib.dump(model_lr, 'models/best_model.pkl')
elif best_model_name == 'RandomForest':
    joblib.dump(model_rf, 'models/best_model.pkl')
elif best_model_name == 'RandomForest_Optimized':
    joblib.dump(model_rf_opt, 'models/best_model.pkl')
elif best_model_name == 'XGBoost':
    joblib.dump(model_xgb, 'models/best_model.pkl')
elif best_model_name == 'XGBoost_Optimized':
    joblib.dump(model_xgb_opt, 'models/best_model.pkl')
elif best_model_name == 'CatBoost':
    joblib.dump(model_cat, 'models/best_model.pkl')
elif best_model_name == 'LightGBM':
    joblib.dump(model_lgb, 'models/best_model.pkl')

best_model_info = {
    'model_name': best_model_name,
    'test_auc': best_model_auc,
    'test_log_loss': best_model_log_loss,
    'test_brier_score': best_model_brier,
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_features': X_train.shape[1],
    'n_train_samples': len(y_train),
    'n_val_samples': len(y_val),
    'n_test_samples': len(y_test)
}

joblib.dump(best_model_info, 'models/best_model_info.pkl')

print("\n" + "="*80)
print("СРАВНЕНИЕ МОДЕЛЕЙ ПО КАЧЕСТВУ И СКОРОСТИ")
print("="*80)
print(results_df.to_string(index=False))

print("\n" + "="*80)
print("РЕКОМЕНДАЦИИ ПО ВЫБОРУ МОДЕЛИ")
print("="*80)

fast_models = results_df[results_df['latency_mean_ms'] < 5]
accurate_models = results_df[results_df['auc'] > 0.7]

print(f"\nБыстрые модели (latency < 5 мс):")
for _, row in fast_models.iterrows():
    print(f"  {row['model']}: AUC={row['auc']:.4f}, latency={row['latency_mean_ms']:.2f}мс")

print(f"\nТочные модели (AUC > 0.7):")
for _, row in accurate_models.iterrows():
    print(f"  {row['model']}: AUC={row['auc']:.4f}, latency={row['latency_mean_ms']:.2f}мс")

print("\n" + "-"*40)
print(f"\nЛучшая модель: {best_model_name}")
print(f"Test AUC: {best_model_auc:.4f}")
print(f"Test Log Loss: {best_model_log_loss:.4f}")
print(f"Test Brier Score: {best_model_brier:.4f}")

preprocessor = joblib.load('preprocessor.pkl')
joblib.dump(preprocessor, 'models/preprocessor.pkl')

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

models = results_df['model'].values
auc_values = results_df['auc'].values

axes[0,0].barh(models, auc_values, color='steelblue')
axes[0,0].set_xlabel('AUC-ROC')
axes[0,0].set_title('Model Comparison by AUC-ROC')
for i, v in enumerate(auc_values):
    axes[0,0].text(v + 0.005, i, f'{v:.4f}', va='center')

log_loss_values = results_df['log_loss'].values
axes[0,1].barh(models, log_loss_values, color='coral')
axes[0,1].set_xlabel('Log Loss')
axes[0,1].set_title('Model Comparison by Log Loss')
for i, v in enumerate(log_loss_values):
    axes[0,1].text(v + 0.01, i, f'{v:.4f}', va='center')

latency_values = results_df['latency_mean_ms'].values
colors = ['green' if x < 5 else 'orange' if x < 20 else 'red' for x in latency_values]
axes[1,0].barh(models, latency_values, color=colors)
axes[1,0].set_xlabel('Latency Mean (ms)')
axes[1,0].set_title('Model Comparison by Inference Speed')
axes[1,0].axvline(x=5, color='green', linestyle='--', alpha=0.7, label='Target (<5ms)')
axes[1,0].axvline(x=20, color='orange', linestyle='--', alpha=0.7, label='Acceptable (<20ms)')
axes[1,0].legend()
for i, v in enumerate(latency_values):
    axes[1,0].text(v + 0.5, i, f'{v:.2f}ms', va='center')

brier_values = results_df['brier_score'].values
axes[1,1].barh(models, brier_values, color='seagreen')
axes[1,1].set_xlabel('Brier Score')
axes[1,1].set_title('Model Comparison by Brier Score')
for i, v in enumerate(brier_values):
    axes[1,1].text(v + 0.002, i, f'{v:.4f}', va='center')

plt.tight_layout()
plt.savefig('experiments_comparison_with_latency.png', dpi=150)
plt.close()

