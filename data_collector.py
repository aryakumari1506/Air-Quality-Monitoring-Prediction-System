import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json

class AirQualityDataCollector:    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.geocoding_url = "http://api.openweathermap.org/geo/1.0/direct"
        
    def get_coordinates(self, city_name):
        try:
            params = {
                'q': city_name,
                'limit': 1,
                'appid': self.api_key
            }
            
            response = requests.get(self.geocoding_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if data:
                return data[0]['lat'], data[0]['lon']
            else:
                print(f"City not found: {city_name}")
                return None
                
        except Exception as e:
            print(f"Error getting coordinates for {city_name}: {str(e)}")
            return None
    
    def get_current_air_quality(self, lat, lon):
        try:
            url = f"{self.base_url}/air_pollution"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            print(f"Error getting current air quality: {str(e)}")
            return None
    
    def get_historical_air_quality(self, lat, lon, start_timestamp, end_timestamp):
        try:
            url = f"{self.base_url}/air_pollution/history"
            params = {
                'lat': lat,
                'lon': lon,
                'start': start_timestamp,
                'end': end_timestamp,
                'appid': self.api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            print(f"Error getting historical air quality: {str(e)}")
            return None
    
    def collect_historical_data(self, city_name, days_back=30):
        coords = self.get_coordinates(city_name)
        if not coords:
            return pd.DataFrame()
        
        lat, lon = coords
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        start_timestamp = int(start_time.timestamp())
        end_timestamp = int(end_time.timestamp())
        
        all_data = []
        chunk_size = 7  # days
        current_start = start_timestamp
        
        while current_start < end_timestamp:
            current_end = min(current_start + (chunk_size * 24 * 3600), end_timestamp)
            
            data = self.get_historical_air_quality(lat, lon, current_start, current_end)
            if data and 'list' in data:
                all_data.extend(data['list'])
            
            current_start = current_end
            time.sleep(0.1)  # Rate limiting
        
        if not all_data:
            print(f"No data collected for {city_name}")
            return pd.DataFrame()
        
        df = self._process_air_quality_data(all_data, city_name)
        
        print(f"Collected {len(df)} records for {city_name}")
        return df
    
    def _process_air_quality_data(self, data_list, city_name):
        processed_data = []
        
        for item in data_list:
            try:
                record = {
                    'city': city_name,
                    'timestamp': pd.to_datetime(item['dt'], unit='s'),
                    'aqi': item['main']['aqi'],
                    'co': item['components']['co'],
                    'no': item['components']['no'],
                    'no2': item['components']['no2'],
                    'o3': item['components']['o3'],
                    'so2': item['components']['so2'],
                    'pm2_5': item['components']['pm2_5'],
                    'pm10': item['components']['pm10'],
                    'nh3': item['components']['nh3']
                }
                processed_data.append(record)
                
            except KeyError as e:
                print(f"Missing key in data: {e}")
                continue
        
        df = pd.DataFrame(processed_data)
        
        if not df.empty:
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            
            df['aqi_category'] = df['aqi'].map({
                1: 'Good',
                2: 'Fair', 
                3: 'Moderate',
                4: 'Poor',
                5: 'Very Poor'
            })
        
        return df
    
    def generate_synthetic_data(self, city_name, days_back=30):        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        
        np.random.seed(hash(city_name) % 2147483647) 
        
        n_points = len(date_range)
        
        base_levels = {
            'Delhi,IN': {'pm2_5': 80, 'pm10': 120, 'no2': 45, 'aqi_base': 4},
            'Beijing,CN': {'pm2_5': 65, 'pm10': 95, 'no2': 40, 'aqi_base': 3},
            'Los Angeles,US': {'pm2_5': 25, 'pm10': 40, 'no2': 30, 'aqi_base': 2},
            'London,UK': {'pm2_5': 20, 'pm10': 35, 'no2': 35, 'aqi_base': 2},
            'Tokyo,JP': {'pm2_5': 15, 'pm10': 25, 'no2': 25, 'aqi_base': 2},
            'Mumbai,IN': {'pm2_5': 70, 'pm10': 110, 'no2': 50, 'aqi_base': 4},
            'Mexico City,MX': {'pm2_5': 45, 'pm10': 70, 'no2': 40, 'aqi_base': 3},
            'São Paulo,BR': {'pm2_5': 35, 'pm10': 55, 'no2': 35, 'aqi_base': 3}
        }
        
        city_base = base_levels.get(city_name, {
            'pm2_5': 30, 'pm10': 50, 'no2': 30, 'aqi_base': 2
        })
        
        data = []
        
        for i, timestamp in enumerate(date_range):
            # Daily cycle 
            hour_factor = 1 + 0.3 * np.sin(2 * np.pi * timestamp.hour / 24 - np.pi/2)
            
            # Weekly cycle
            week_factor = 1.2 if timestamp.weekday() < 5 else 0.8
            
            # Seasonal trend
            seasonal_factor = 1 + 0.2 * np.cos(2 * np.pi * timestamp.month / 12)
            
            # Random noise
            noise = np.random.normal(0, 0.15)
            
            combined_factor = hour_factor * week_factor * seasonal_factor * (1 + noise)
            
            record = {
                'city': city_name,
                'timestamp': timestamp,
                'pm2_5': max(1, city_base['pm2_5'] * combined_factor),
                'pm10': max(1, city_base['pm10'] * combined_factor),
                'no2': max(1, city_base['no2'] * combined_factor),
                'so2': max(1, 20 * combined_factor),
                'co': max(100, 1000 * combined_factor),
                'o3': max(1, 80 * combined_factor),
                'nh3': max(1, 10 * combined_factor),
                'no': max(1, 15 * combined_factor)
            }
            
            # Calculate AQI based on PM2.5
            pm2_5_aqi = min(5, max(1, int(record['pm2_5'] / 15) + 1))
            record['aqi'] = pm2_5_aqi
            
            data.append(record)
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        # Add derived features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        
        # Calculate AQI categories
        df['aqi_category'] = df['aqi'].map({
            1: 'Good',
            2: 'Fair',
            3: 'Moderate', 
            4: 'Poor',
            5: 'Very Poor'
        })
        
        return df