"""
Configuration file for Air Quality Monitoring System
"""

import os

class Config:
    """Configuration class containing all system settings"""
    
    def __init__(self):
        # API Configuration
        self.api_key = os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.air_pollution_endpoint = "/air_pollution"
        self.history_endpoint = "/air_pollution/history"
        
        # Default cities for analysis
        self.default_cities = [
            "Delhi,IN",
            "Beijing,CN",
            "Los Angeles,US",
            "London,UK",
            "Tokyo,JP",
            "Mumbai,IN",
            "Mexico City,MX",
            "São Paulo,BR"
        ]
        
        # WHO Air Quality Guidelines (μg/m³)
        self.who_limits = {
            'pm2_5': 15,    # PM2.5 annual mean
            'pm10': 45,     # PM10 annual mean
            'no2': 40,      # NO2 annual mean
            'so2': 20,      # SO2 daily mean
            'co': 30000,    # CO 8-hour mean (converted to μg/m³)
            'o3': 100       # O3 8-hour daily maximum mean
        }
        
        # AQI Thresholds
        self.aqi_thresholds = {
            1: {'label': 'Good', 'color': '#00E400'},
            2: {'label': 'Fair', 'color': '#FFFF00'},
            3: {'label': 'Moderate', 'color': '#FF7E00'},
            4: {'label': 'Poor', 'color': '#FF0000'},
            5: {'label': 'Very Poor', 'color': '#8F3F97'}
        }
        
        # Population data for risk scoring (millions)
        self.city_populations = {
            'Delhi,IN': 30.29,
            'Beijing,CN': 21.54,
            'Los Angeles,US': 3.97,
            'London,UK': 9.54,
            'Tokyo,JP': 13.96,
            'Mumbai,IN': 20.41,
            'Mexico City,MX': 9.21,
            'São Paulo,BR': 12.33
        }
        
        # ARIMA model parameters
        self.arima_params = {
            'max_p': 5,
            'max_d': 2,
            'max_q': 5,
            'seasonal': True,
            'seasonal_periods': 7  # Weekly seasonality
        }
        
        # Visualization settings
        self.viz_settings = {
            'figsize': (15, 10),
            'style': 'seaborn-v0_8',
            'color_palette': 'viridis',
            'dpi': 300
        }
        
        # Output directories
        self.output_dirs = {
            'plots': 'outputs/plots',
            'reports': 'outputs/reports',
            'data': 'outputs/data'
        }