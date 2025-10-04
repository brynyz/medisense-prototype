import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import numpy as np

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
url = "https://air-quality-api.open-meteo.com/v1/air-quality"
params = {
    "latitude": 16.9375,
    "longitude": 121.7645,
    "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "ozone"],
    "start_date": "2022-06-01",
    "end_date": "2025-09-20",
    "timezone": "Asia/Manila"
}

responses = openmeteo.weather_api(url, params=params)

# Process first location
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone: Asia/Manila")

# Process hourly data
hourly = response.Hourly()
hourly_pm10 = hourly.Variables(0).ValuesAsNumpy()
hourly_pm2_5 = hourly.Variables(1).ValuesAsNumpy()
hourly_carbon_monoxide = hourly.Variables(2).ValuesAsNumpy()
hourly_nitrogen_dioxide = hourly.Variables(3).ValuesAsNumpy()
hourly_ozone = hourly.Variables(4).ValuesAsNumpy()

# Create hourly dataframe first
hourly_data = {
    "datetime": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    ).tz_convert('Asia/Manila'),
    "pm10": hourly_pm10,
    "pm2_5": hourly_pm2_5,
    "carbon_monoxide": hourly_carbon_monoxide,
    "nitrogen_dioxide": hourly_nitrogen_dioxide,
    "ozone": hourly_ozone
}

hourly_df = pd.DataFrame(data = hourly_data)

# Convert to daily means
hourly_df['date'] = hourly_df['datetime'].dt.date

daily_aqi = hourly_df.groupby('date').agg({
    'pm10': ['mean', 'max', 'min'],
    'pm2_5': ['mean', 'max', 'min'], 
    'carbon_monoxide': ['mean', 'max', 'min'],
    'nitrogen_dioxide': ['mean', 'max', 'min'],
    'ozone': ['mean', 'max', 'min']
}).round(2)

# Flatten column names
daily_aqi.columns = [f"{col[0]}_{col[1]}" for col in daily_aqi.columns]
daily_aqi = daily_aqi.reset_index()

# Add health-relevant indicators
def get_aqi_category(pm2_5_mean):
    """Simplified AQI based on PM2.5 (WHO guidelines)"""
    if pd.isna(pm2_5_mean):
        return 'unknown'
    elif pm2_5_mean <= 15:
        return 'good'
    elif pm2_5_mean <= 25:
        return 'moderate'
    elif pm2_5_mean <= 50:
        return 'unhealthy_sensitive'
    elif pm2_5_mean <= 75:
        return 'unhealthy'
    else:
        return 'hazardous'

daily_aqi['aqi_category'] = daily_aqi['pm2_5_mean'].apply(get_aqi_category)

# Health risk flags
daily_aqi['high_pm2_5'] = daily_aqi['pm2_5_mean'] > 25  # WHO annual guideline
daily_aqi['high_pm10'] = daily_aqi['pm10_mean'] > 50   # WHO annual guideline
daily_aqi['high_ozone'] = daily_aqi['ozone_mean'] > 100 # General threshold

# Respiratory risk score (0-3)
daily_aqi['respiratory_risk_score'] = (
    daily_aqi['high_pm2_5'].astype(int) + 
    daily_aqi['high_pm10'].astype(int) + 
    daily_aqi['high_ozone'].astype(int)
)

# 7-day rolling averages for trend analysis
for pollutant in ['pm10_mean', 'pm2_5_mean', 'ozone_mean']:
    daily_aqi[f'{pollutant}_7day'] = daily_aqi[pollutant].rolling(window=7, center=True).mean().round(2)

print(f"\n✅ Daily AQI Summary:")
print(f"Total days: {len(daily_aqi)}")
print(f"Date range: {daily_aqi['date'].min()} to {daily_aqi['date'].max()}")

print(f"\nAQI Category Distribution:")
print(daily_aqi['aqi_category'].value_counts())

print(f"\nAverage Pollutant Levels:")
print(f"PM2.5: {daily_aqi['pm2_5_mean'].mean():.1f} μg/m³")
print(f"PM10: {daily_aqi['pm10_mean'].mean():.1f} μg/m³") 
print(f"Ozone: {daily_aqi['ozone_mean'].mean():.1f} μg/m³")

print(f"\nHigh Pollution Days:")
print(f"High PM2.5: {daily_aqi['high_pm2_5'].sum()} days ({daily_aqi['high_pm2_5'].mean()*100:.1f}%)")
print(f"High PM10: {daily_aqi['high_pm10'].sum()} days ({daily_aqi['high_pm10'].mean()*100:.1f}%)")
print(f"High Ozone: {daily_aqi['high_ozone'].sum()} days ({daily_aqi['high_ozone'].mean()*100:.1f}%)")

print(f"\nDaily AQI data preview:")
print(daily_aqi.head())

# Save daily AQI data
daily_aqi.to_csv('medisense/backend/data/environment/daily_aqi.csv', index=False)
print(f"\n💾 Saved: medisense/backend/data/environment/daily_aqi.csv")
print(f"Ready for merging with medical data! 🎯")