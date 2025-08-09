#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from data_collector import AirQualityDataCollector
from analyzer import AirQualityAnalyzer
from predictor import ARIMAPredictor
from risk_assessor import RiskAssessor
from visualizer import DataVisualizer
from config import Config

class AirQualityMonitoringSystem:    
    def __init__(self):
        self.config = Config()
        self.data_collector = AirQualityDataCollector(self.config.api_key)
        self.analyzer = AirQualityAnalyzer()
        self.predictor = ARIMAPredictor()
        self.risk_assessor = RiskAssessor()
        self.visualizer = DataVisualizer()
        
    def run_analysis(self, cities=None, days_back=30):
        if cities is None:
            cities = self.config.default_cities
            
        print("="*60)
        
        # Step 1: Data Collection
        all_data = {}
        
        for city in cities:
            data = self.data_collector.collect_historical_data(city, days_back)
            if not data.empty:
                all_data[city] = data
                print(f"  Collected {len(data)} records for {city}")
        
        if not all_data:
            return
        
        # Step 2: Data Analysis
        analysis_results = {}
        
        for city, data in all_data.items():
            print(f"  Analyzing {city}...")
            
            # Seasonal decomposition
            decomposition = self.analyzer.seasonal_decomposition(data)
            
            # Correlation analysis
            correlations = self.analyzer.correlation_analysis(data)
            
            # Statistical summary
            stats = self.analyzer.statistical_summary(data)
            
            analysis_results[city] = {
                'data': data,
                'decomposition': decomposition,
                'correlations': correlations,
                'statistics': stats
            }
            
            print(f"  Analysis completed for {city}")
        
        # Step 3: Prediction
        predictions = {}
        
        for city, results in analysis_results.items():
            print(f"  Predicting AQI for {city}...")
            
            try:
                forecast, mae = self.predictor.forecast_aqi(
                    results['data'], 
                    forecast_days=7
                )
                predictions[city] = {
                    'forecast': forecast,
                    'mae': mae
                }
                print(f"  Prediction completed for {city} (MAE: {mae:.2f})")
            except Exception as e:
                print(f"  Prediction failed for {city}: {str(e)}")
                predictions[city] = None
        
        # Step 4: Risk Assessment
        risk_results = {}
        
        for city, results in analysis_results.items():
            print(f"  Assessing risk for {city}...")
            
            risk_metrics = self.risk_assessor.assess_city_risk(
                results['data'], 
                city
            )
            risk_results[city] = risk_metrics
            
            exceedance_factor = risk_metrics['exceedance_factor']
            risk_score = risk_metrics['risk_score']
            
            print(f"  {city} - Exceedance Factor: {exceedance_factor:.2f}x, Risk Score: {risk_score:.2f}")
        
        # Step 5: Visualization
        self.visualizer.create_comprehensive_dashboard(
            analysis_results, 
            predictions, 
            risk_results
        )
        
        # Step 6: Generate Report
        self._generate_summary_report(analysis_results, predictions, risk_results)
            
    def _generate_summary_report(self, analysis_results, predictions, risk_results):        
        report = []
        report.append("AIR QUALITY MONITORING & PREDICTION SYSTEM")
        report.append("="*50)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # High-risk cities identification
        high_risk_cities = []
        for city, risk in risk_results.items():
            if risk['exceedance_factor'] > 2.0:  # Cities exceeding WHO limits by 2x
                high_risk_cities.append((city, risk['exceedance_factor']))
        
        high_risk_cities.sort(key=lambda x: x[1], reverse=True)
        
        report.append(" HIGH-RISK CITIES (Exceeding WHO Limits by 2x+):")
        report.append("-" * 45)
        
        if high_risk_cities:
            avg_exceedance = np.mean([factor for _, factor in high_risk_cities])
            report.append(f"Found {len(high_risk_cities)} high-risk cities")
            report.append(f"Average exceedance factor: {avg_exceedance:.1f}x WHO limits")
            report.append("")
            
            for city, factor in high_risk_cities[:5]:  # Top 5
                report.append(f"  • {city}: {factor:.2f}x WHO limits")
        else:
            report.append("  No cities found exceeding WHO limits by 2x")
        
        report.append("")
        
        # Prediction accuracy summary
        report.append(" PREDICTION ACCURACY:")
        report.append("-" * 25)
        
        mae_values = []
        for city, pred in predictions.items():
            if pred:
                mae_values.append(pred['mae'])
                report.append(f"  • {city}: MAE = {pred['mae']:.2f}")
        
        if mae_values:
            avg_mae = np.mean(mae_values)
            report.append(f"\n  Average MAE across all cities: {avg_mae:.2f}")
        
        report.append("")
        
        # City statistics
        report.append("CITY STATISTICS:")
        report.append("-" * 18)
        
        for city, results in analysis_results.items():
            stats = results['statistics']
            risk = risk_results[city]
            
            report.append(f"\n{city}:")
            report.append(f"  Average AQI: {stats['aqi_mean']:.1f}")
            report.append(f"  Max AQI: {stats['aqi_max']:.1f}")
            report.append(f"  Risk Score: {risk['risk_score']:.2f}")
            report.append(f"  Days exceeding WHO limits: {risk['days_exceeding_who']}")
        
        os.makedirs('outputs', exist_ok=True)
        with open('outputs/summary_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print('\n'.join(report))

def main():
    system = AirQualityMonitoringSystem()
    
    cities = [
        "Delhi,IN",
        "Beijing,CN", 
        "Los Angeles,US",
        "London,UK",
        "Tokyo,JP",
        "Mumbai,IN",
        "Mexico City,MX",
        "São Paulo,BR"
    ]
    
if __name__ == "__main__":
    main()