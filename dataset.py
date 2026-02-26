# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

taxi = pd.read_csv('./data/spb_taxi_dataset.csv')
taxi.columns = ['trip_start', 'start_coord', 'end_coord', 'trip_length',
                'weather', 'time_of_day', 'day_of_week', 'demand',
                'cost', 'start_lat', 'start_lon', 'end_lat', 'end_lon']
print(f"Taxi поездок: {taxi.shape}")

casco = pd.read_csv('./data/casco_train.csv')
casco_cols = {
    'column_1': 'gender',
    'column_2': 'driving_experience',
    'column_3': 'marital_status',
    'column_4': 'driver_city',
    'column_5': 'contract_city',
    'column_6': 'max_payment_delay',
    'column_7': 'engine_type',
    'column_8': 'vehicle_mileage',
    'column_9': 'vehicle_year',
    'column_10': 'engine_power',
    'column_11': 'num_client_contracts',
    'column_12': 'num_vehicle_contracts',
    'column_13': 'age',
    'column_14': 'vehicle_type',
    'column_15': 'manufacturing_country',
    'column_16': 'has_telematics',
    'column_17': 'brand_model',
    'column_18': 'technical_param'
}
casco = casco.rename(columns=casco_cols)
print(f"CASCO профилей: {casco.shape}")

telemetry = pd.read_csv('./data/Driver_Behavior.csv')
print(f"Телематика записей: {telemetry.shape}")
print(f"Колонки телематики: {list(telemetry.columns)}")

fraud = pd.read_csv('./data/fraud_detection.csv')
print(f"Fraud записей: {fraud.shape}")



driver_pool = casco[['age', 'driving_experience', 'gender', 'marital_status',
                     'max_payment_delay', 'num_client_contracts']].dropna().copy()
driver_pool['driving_experience'] = driver_pool.apply(
    lambda x: min(x['driving_experience'], x['age'] - 18) if x['age'] > 18 else 0,
    axis=1
)
driver_pool = driver_pool.sample(n=2000, replace=True, random_state=RANDOM_SEED).reset_index(drop=True)
driver_pool.index.name = 'driver_id'
print(f"Создан пул из {len(driver_pool)} виртуальных водителей")

behavior_stats = telemetry.groupby('behavior_label').agg({
    'speed_kmph': ['mean', 'std'],
    'accel_x': ['mean', 'std'],
    'accel_y': ['mean', 'std'],
    'brake_pressure': ['mean', 'std'],
    'steering_angle': ['mean', 'std'],
    'throttle': ['mean', 'std'],
    'lane_deviation': ['mean', 'std'],
    'phone_usage': ['mean', 'std'],
    'headway_distance': ['mean', 'std'],
    'reaction_time': ['mean', 'std']
}).round(4)
print("\nСтатистика телематики по стилям вождения:")
print(behavior_stats)

fraud['accident'] = fraud['FraudFound_P']
fraud['age_group'] = pd.cut(fraud['Age'], bins=[18, 25, 35, 50, 100],
                             labels=['18-25', '26-35', '36-50', '50+'])
age_risk = fraud.groupby('age_group')['accident'].mean().to_dict()
print("\nРиск ДТП по возрастным группам (из Fraud):")
for age, risk in age_risk.items():
    print(f"  {age}: {risk:.4f}")

gender_risk = fraud.groupby('Sex')['accident'].mean().to_dict()
print(f"\nРиск по полу: {gender_risk}")

fault_risk = fraud.groupby('Fault')['accident'].mean().to_dict()
print(f"Риск по Fault: {fault_risk}")


base_trips = taxi.sample(n=50000, replace=True, random_state=RANDOM_SEED).copy().reset_index(drop=True)

base_trips['trip_start'] = pd.to_datetime(base_trips['trip_start'], errors='coerce', dayfirst=True)
initial_len = len(base_trips)
base_trips = base_trips.dropna(subset=['trip_start'])
print(f"Удалено {initial_len - len(base_trips)} строк с некорректной датой")

base_trips['hour'] = base_trips['trip_start'].dt.hour
base_trips['month'] = base_trips['trip_start'].dt.month
base_trips['is_weekend'] = base_trips['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
base_trips['is_night'] = ((base_trips['hour'] >= 22) | (base_trips['hour'] <= 5)).astype(int)

weather_map = {'солнечно': 'Clear', 'облачно': 'Cloudy', 'дождь': 'Rain', 'снег': 'Snow'}
base_trips['weather'] = base_trips['weather'].map(weather_map).fillna('Clear')

if 'trip_duration' not in base_trips.columns:
    base_trips['trip_duration_min'] = (base_trips['trip_length'] / 25 * 60).clip(lower=5, upper=180)
else:
    base_trips['trip_duration_min'] = base_trips['trip_duration']

print(f"База поездок после обработки: {base_trips.shape}")
print(f"Колонки: {list(base_trips.columns)}")



# Назначаем водителя каждой поездке
base_trips['driver_id'] = np.random.choice(driver_pool.index, size=len(base_trips))
driver_features = driver_pool.add_prefix('driver_')
base_trips = base_trips.join(driver_features, on='driver_id')

# Добавляем случайные характеристики авто из CASCO
vehicle_cols = ['vehicle_mileage', 'vehicle_year', 'engine_power',
                'engine_type', 'vehicle_type', 'manufacturing_country']
for col in vehicle_cols:
    if col in casco.columns:
        valid_vals = casco[col].dropna()
        base_trips[f'vehicle_{col}'] = np.random.choice(valid_vals, size=len(base_trips), replace=True)

if 'hour' not in base_trips.columns:
    base_trips['hour'] = base_trips['trip_start'].dt.hour
if 'is_night' not in base_trips.columns:
    base_trips['is_night'] = ((base_trips['hour'] >= 22) | (base_trips['hour'] <= 5)).astype(int)
    print("Колонка 'is_night' восстановлена.")

print("Колонки после добавления всех признаков:", base_trips.columns.tolist())

def determine_behavior(row):
    age = row['driver_age']
    is_night = row['is_night']
    weather = row['weather']
    
    if age < 25 and is_night and weather in ['Rain', 'Snow']:
        return np.random.choice(['Aggressive', 'Distracted', 'Safe'], p=[0.6, 0.3, 0.1])
    elif age > 60 and is_night:
        return np.random.choice(['Safe', 'Aggressive', 'Distracted'], p=[0.8, 0.1, 0.1])
    elif weather == 'Clear' and not is_night:
        return np.random.choice(['Safe', 'Aggressive', 'Distracted'], p=[0.5, 0.25, 0.25])
    elif weather in ['Rain', 'Snow']:
        return np.random.choice(['Safe', 'Aggressive', 'Distracted'], p=[0.5, 0.2, 0.3])
    else:
        return np.random.choice(['Safe', 'Aggressive', 'Distracted'], p=[0.4, 0.3, 0.3])

def generate_telemetry_for_trip(row, behavior_stats):
    behavior = row['behavior']
    stats = behavior_stats.loc[behavior] 
    duration = row['trip_duration_min'] / 30  
    
    telemetry_data = {
        'driving_style': behavior,
        'avg_speed': np.random.normal(stats[('speed_kmph', 'mean')],
                                     stats[('speed_kmph', 'std')]),
        'avg_accel_x': np.random.normal(stats[('accel_x', 'mean')],
                                       stats[('accel_x', 'std')]),
        'avg_accel_y': np.random.normal(stats[('accel_y', 'mean')],
                                       stats[('accel_y', 'std')]),
        'avg_brake_pressure': np.random.normal(stats[('brake_pressure', 'mean')],
                                              stats[('brake_pressure', 'std')]),
        'steering_variability': abs(np.random.normal(stats[('steering_angle', 'mean')],
                                                    stats[('steering_angle', 'std')])),
        'avg_throttle': np.random.normal(stats[('throttle', 'mean')],
                                        stats[('throttle', 'std')]),
        'lane_deviation': abs(np.random.normal(stats[('lane_deviation', 'mean')],
                                              stats[('lane_deviation', 'std')])),
        'phone_usage_prob': min(1, max(0, np.random.normal(stats[('phone_usage', 'mean')],
                                                          stats[('phone_usage', 'std')]))),
        'headway_distance': max(0, np.random.normal(stats[('headway_distance', 'mean')],
                                                   stats[('headway_distance', 'std')])),
        'reaction_time': max(0.1, np.random.normal(stats[('reaction_time', 'mean')],
                                                  stats[('reaction_time', 'std')]))
    }
    
    telemetry_data['hard_brakes_count'] = int(
        max(0, telemetry_data['avg_brake_pressure'] * duration * np.random.uniform(0.5, 1.5))
    )
    telemetry_data['hard_accelerations_count'] = int(
        max(0, telemetry_data['avg_throttle'] * duration * np.random.uniform(0.5, 1.5))
    )
    telemetry_data['sharp_turns_count'] = int(
        max(0, telemetry_data['steering_variability'] * duration * np.random.uniform(0.3, 1.2))
    )
    
    return pd.Series(telemetry_data)

base_trips['behavior'] = base_trips.apply(determine_behavior, axis=1)

# Генерируем телематику
telemetry_features = base_trips.apply(
    lambda row: generate_telemetry_for_trip(row, behavior_stats), axis=1
)
base_trips = pd.concat([base_trips, telemetry_features], axis=1)
print("Телематика сгенерирована")

def calculate_accident_probability(row):
    prob = 0.01  # базовый риск 1%
    age = row['driver_age']
    if age < 25:
        prob *= 2.0
    elif age < 35:
        prob *= 1.3
    elif age < 50:
        prob *= 1.0
    else:
        prob *= 0.8
    
    if row.get('driver_gender') == 'M':
        prob *= 1.2
    
    style = row['driving_style']
    if style == 'Aggressive':
        prob *= 2.5
    elif style == 'Distracted':
        prob *= 3.0
    else:
        prob *= 0.6
    
    weather = row['weather']
    if weather == 'Rain':
        prob *= 1.5
    elif weather == 'Snow':
        prob *= 2.2
    
    if row['is_night']:
        prob *= 1.4
    
    if row.get('driver_driving_experience', 0) < 2:
        prob *= 1.4
    
    if row.get('driver_max_payment_delay', 0) > 30:
        prob *= 1.3
    
    if row['hard_brakes_count'] > 5:
        prob *= 1.3
    if row['phone_usage_prob'] > 0.5:
        prob *= 1.5
    
    return min(prob, 0.3)

base_trips['accident_prob'] = base_trips.apply(calculate_accident_probability, axis=1)
base_trips['is_accident'] = (np.random.random(len(base_trips)) < base_trips['accident_prob']).astype(int)

print(f"Сгенерировано ДТП: {base_trips['is_accident'].sum()} из {len(base_trips)} "
      f"({base_trips['is_accident'].mean()*100:.2f}%)")


final_columns = {
    'trip_start': 'trip_start_datetime',
    'trip_length': 'trip_distance_km',
    'trip_duration_min': 'trip_duration_min',
    'weather': 'weather_condition',
    'time_of_day': 'time_of_day_category',
    'day_of_week': 'day_of_week',
    'is_weekend': 'is_weekend',
    'is_night': 'is_night_drive',
    'demand': 'demand_coefficient',
    'cost': 'trip_cost_rub',
    'start_lat': 'start_latitude',
    'start_lon': 'start_longitude',
    'end_lat': 'end_latitude',
    'end_lon': 'end_longitude',
    'driver_age': 'driver_age',
    'driver_driving_experience': 'driving_experience_years',
    'driver_gender': 'gender',
    'driver_marital_status': 'marital_status',
    'driver_max_payment_delay': 'max_payment_delay_days',
    'driver_num_client_contracts': 'num_prev_contracts',
    'vehicle_vehicle_mileage': 'vehicle_mileage_km',
    'vehicle_vehicle_year': 'vehicle_manufacture_year',
    'vehicle_engine_power': 'engine_horsepower',
    'vehicle_engine_type': 'engine_type',
    'vehicle_vehicle_type': 'vehicle_category',
    'vehicle_manufacturing_country': 'manufacturing_country',
    'driving_style': 'driving_style',
    'avg_speed': 'avg_speed_kmh',
    'avg_accel_x': 'avg_longitudinal_accel',
    'avg_accel_y': 'avg_lateral_accel',
    'avg_brake_pressure': 'avg_brake_pressure',
    'steering_variability': 'steering_variability',
    'avg_throttle': 'avg_throttle_pos',
    'lane_deviation': 'lane_deviation',
    'phone_usage_prob': 'phone_usage_probability',
    'headway_distance': 'avg_headway_distance',
    'reaction_time': 'avg_reaction_time_sec',
    'hard_brakes_count': 'hard_brakes_count',
    'hard_accelerations_count': 'hard_accelerations_count',
    'sharp_turns_count': 'sharp_turns_count',
    'accident_prob': 'accident_probability',
    'is_accident': 'is_accident'
}

available_cols = {k: v for k, v in final_columns.items() if k in base_trips.columns}
final_df = base_trips[list(available_cols.keys())].rename(columns=available_cols)
final_df = final_df.loc[:, ~final_df.columns.duplicated()].dropna()

print(f"Финальный датасет: {final_df.shape}")
print(f"Колонки: {list(final_df.columns)}")

final_df.to_csv('car_sharing_insurance_complete.csv', index=False)


print("\n Распределение целевой переменной")
print(final_df['is_accident'].value_counts(normalize=True))

print("\n Распределение стилей вождения ")
print(final_df['driving_style'].value_counts(normalize=True))

print("\nСредние показатели телематики по стилям вождения ")
telemetry_cols = ['avg_speed_kmh', 'hard_brakes_count', 'hard_accelerations_count',
                  'sharp_turns_count', 'phone_usage_probability', 'lane_deviation']
print(final_df.groupby('driving_style')[telemetry_cols].mean().round(2))

print("\n Частота ДТП по погодным условиям ")
print(final_df.groupby('weather_condition')['is_accident'].mean().sort_values(ascending=False))

print("\nЧастота ДТП по возрастным группам ")
final_df['age_group'] = pd.cut(final_df['driver_age'],
                               bins=[18, 25, 35, 50, 100],
                               labels=['18-25', '26-35', '36-50', '50+'])
print(final_df.groupby('age_group')['is_accident'].mean().sort_values(ascending=False))

print("\nКорреляция числовых признаков с is_accident (топ-10)")
numeric_cols = final_df.select_dtypes(include=[np.number]).columns
corr_with_target = final_df[numeric_cols].corr()['is_accident'].drop('is_accident').abs().sort_values(ascending=False).head(10)
print(corr_with_target)

