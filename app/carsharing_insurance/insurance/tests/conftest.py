# insurance/tests/conftest.py
import pytest
from django.test import Client
from django.contrib.auth.models import User

@pytest.fixture
def client():
    return Client()

@pytest.fixture
def sample_trip_data():
    return {
        'trip_distance_km': 5.2,
        'trip_duration_min': 15.0,
        'trip_cost_rub': 250.0,
        'driver_age': 25,
        'driving_experience_years': 3,
        'hour': 22,
        'day_of_week_num': 5,
        'is_weekend': 1,
        'weather_condition': 'Rain',
        'driving_style': 'Aggressive',
        'gender': 'M',
        'hard_brakes_count': 3,
        'hard_accelerations_count': 5,
        'sharp_turns_count': 2
    }

@pytest.fixture
def low_risk_trip_data():
    return {
        'trip_distance_km': 5.0,
        'trip_duration_min': 15.0,
        'trip_cost_rub': 180.0,
        'driver_age': 45,
        'driving_experience_years': 20,
        'hour': 14,
        'day_of_week_num': 2,
        'is_weekend': 0,
        'weather_condition': 'Clear',
        'driving_style': 'Safe',
        'gender': 'F',
        'hard_brakes_count': 0,
        'hard_accelerations_count': 0,
        'sharp_turns_count': 0
    }