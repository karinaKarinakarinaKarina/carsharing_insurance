"""
Шаг 1. Подготовка данных для обучения модели
Используется предварительно сгенерированный датасет car_sharing_insurance_complete.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('car_sharing_insurance_complete.csv')
df['trip_start_datetime'] = pd.to_datetime(df['trip_start_datetime'])
df = df.sort_values('trip_start_datetime').reset_index(drop=True)

print(f"Всего записей: {len(df)}")
print(f"Период данных: {df['trip_start_datetime'].min()} - {df['trip_start_datetime'].max()}")
print(f"Количество дней: {df['trip_start_datetime'].dt.date.nunique()}")
print(f"Доля ДТП: {df['is_accident'].mean()*100:.2f}%")
print(f"Количество ДТП: {df['is_accident'].sum()}")


df['hour'] = df['trip_start_datetime'].dt.hour
df['day_of_week_num'] = df['trip_start_datetime'].dt.dayofweek
df['is_weekend'] = (df['day_of_week_num'] >= 5).astype(int)
df['day_of_month'] = df['trip_start_datetime'].dt.day

def time_of_day_group(hour):
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 22:
        return 'evening'
    else:
        return 'night'

df['time_of_day_group'] = df['hour'].apply(time_of_day_group)

def age_group(age):
    if age < 25:
        return 'young'
    elif age < 35:
        return 'young_adult'
    elif age < 50:
        return 'adult'
    else:
        return 'senior'

df['age_group'] = df['driver_age'].apply(age_group)

df['experience_ratio'] = (df['driving_experience_years'] / 
                           (df['driver_age'] - 17).clip(lower=1))

df['hard_events_per_min'] = (
    df['hard_brakes_count'] + 
    df['hard_accelerations_count'] + 
    df['sharp_turns_count']
) / df['trip_duration_min'].clip(lower=1)

df['night_and_inexperienced'] = ((df['is_night_drive'] == 1) & 
                                   (df['driving_experience_years'] < 2)).astype(int)

df['style_weather_risk'] = df['driving_style'].map({
    'Safe': 1,
    'Aggressive': 3,
    'Distracted': 4
}) * df['weather_condition'].map({
    'Clear': 1,
    'Cloudy': 1.2,
    'Rain': 1.5,
    'Snow': 2.0
}).fillna(1)

print(f"  - hour, day_of_week_num, is_weekend, day_of_month")
print(f"  - time_of_day_group, age_group")
print(f"  - experience_ratio, hard_events_per_min")
print(f"  - night_and_inexperienced, style_weather_risk")

# Целевая переменная
TARGET = 'is_accident'

EXCLUDE_COLS = [
    TARGET,                         # целевая переменная
    'accident_probability',         # прямой прокси цели
    'trip_start_datetime',          # временная метка
    'driver_id',                    # идентификатор
]

NUMERIC_FEATURES = [
    'trip_distance_km',
    'trip_duration_min',
    'demand_coefficient',
    'trip_cost_rub',
    
    'driver_age',
    'driving_experience_years',
    'max_payment_delay_days',
    'num_prev_contracts',
    
    'vehicle_mileage_km',
    'vehicle_manufacture_year',
    'engine_horsepower',
    
    'avg_speed_kmh',
    'avg_longitudinal_accel',
    'avg_lateral_accel',
    'avg_brake_pressure',
    'steering_variability',
    'avg_throttle_pos',
    'lane_deviation',
    'phone_usage_probability',
    'avg_headway_distance',
    'avg_reaction_time_sec',
    
    'hard_brakes_count',
    'hard_accelerations_count',
    'sharp_turns_count',
    
    'hour',
    'day_of_week_num',
    'is_weekend',
    'day_of_month',
    
    'experience_ratio',
    'hard_events_per_min',
    'night_and_inexperienced',
    'style_weather_risk'
]

CATEGORICAL_FEATURES = [
    'weather_condition',
    'gender',
    'marital_status',
    'engine_type',
    'vehicle_category',
    'manufacturing_country',
    'driving_style',
    'time_of_day_group',
    'age_group'
]

available_numeric = [f for f in NUMERIC_FEATURES if f in df.columns]
available_categorical = [f for f in CATEGORICAL_FEATURES if f in df.columns]

print(f"\nДоступные числовые признаки: {len(available_numeric)}")
print(f"Доступные категориальные признаки: {len(available_categorical)}")
print(f"Всего признаков: {len(available_numeric) + len(available_categorical)}")


# Разделение на train (80% старых) и test (20% новых)
split_date = df['trip_start_datetime'].quantile(0.8)
train_df = df[df['trip_start_datetime'] <= split_date].copy()
test_df = df[df['trip_start_datetime'] > split_date].copy()

print(f"Дата разделения: {split_date.date()}")
print(f"\nTrain: {len(train_df)} записей")
print(f"  Период: {train_df['trip_start_datetime'].min().date()} - {train_df['trip_start_datetime'].max().date()}")
print(f"  Доля ДТП: {train_df[TARGET].mean()*100:.2f}%")
print(f"\nTest: {len(test_df)} записей")
print(f"  Период: {test_df['trip_start_datetime'].min().date()} - {test_df['trip_start_datetime'].max().date()}")
print(f"  Доля ДТП: {test_df[TARGET].mean()*100:.2f}%")

# Из train выделяем validation с помощью TimeSeriesSplit (последний сплит)
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(train_df):
    pass  

val_df = train_df.iloc[val_idx].copy()
train_final_df = train_df.iloc[train_idx].copy()

print(f"\nValidation (из последнего сплита TimeSeriesSplit):")
print(f"  Период: {val_df['trip_start_datetime'].min().date()} - {val_df['trip_start_datetime'].max().date()}")
print(f"  Записей: {len(val_df)}")
print(f"  Доля ДТП: {val_df[TARGET].mean()*100:.2f}%")
print(f"\nФинальный Train:")
print(f"  Период: {train_final_df['trip_start_datetime'].min().date()} - {train_final_df['trip_start_datetime'].max().date()}")
print(f"  Записей: {len(train_final_df)}")
print(f"  Доля ДТП: {train_final_df[TARGET].mean()*100:.2f}%")

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, available_numeric),
        ('cat', categorical_transformer, available_categorical)
    ]
)

# Обучаем препроцессор ТОЛЬКО на train
X_train = preprocessor.fit_transform(train_final_df[available_numeric + available_categorical])
X_val = preprocessor.transform(val_df[available_numeric + available_categorical])
X_test = preprocessor.transform(test_df[available_numeric + available_categorical])

y_train = train_final_df[TARGET].values
y_val = val_df[TARGET].values
y_test = test_df[TARGET].values

print(f"Размерность X_train: {X_train.shape}")
print(f"Размерность X_val: {X_val.shape}")
print(f"Размерность X_test: {X_test.shape}")
print(f"\nДисбаланс классов в train: {y_train.sum()} ДТП из {len(y_train)} ({y_train.mean()*100:.2f}%)")
print(f"Дисбаланс классов в val: {y_val.sum()} ДТП из {len(y_val)} ({y_val.mean()*100:.2f}%)")
print(f"Дисбаланс классов в test: {y_test.sum()} ДТП из {len(y_test)} ({y_test.mean()*100:.2f}%)")

# Сохраняем препроцессор для использования в инференсе
joblib.dump(preprocessor, 'preprocessor.pkl')

np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)
np.save('y_test.npy', y_test)

feature_info = {
    'numeric_features': available_numeric,
    'categorical_features': available_categorical,
    'n_features': X_train.shape[1],
    'n_train': len(y_train),
    'n_val': len(y_val),
    'n_test': len(y_test)
}
joblib.dump(feature_info, 'feature_info.pkl')