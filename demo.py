#!/usr/bin/env python3
"""
Demo script for Air Quality Monitoring System
Uses synthetic data when API key is not available
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')
from config import Config
from data_collector import AirQualityDataCollector
from analyzer import AirQualityAnalyzer
from predictor import ARIMAPredictor
from risk_assessor import RiskAssessor
from visualizer import DataVisualizer

class AirQualityDemo:
    
    def __init__(self, use_synthetic=False):
        self.config = Config()
        self.use_synthetic = use_synthetic
        
        api_key = os.getenv('OPENWEATHER_API_KEY', 'demo_key')
        self.data_collector = AirQualityDataCollector(api_key)
        self.analyzer = AirQualityAnalyzer()
        self.predictor = ARIMAPredictor()
        self.risk_assessor = RiskAssessor()
        self.visualizer = DataVisualizer()
    
    def run_demo(self, cities=None, days_back=30):        
        if cities is None:
            cities = [
                "Delhi,IN",
                "Beijing,CN", 
                "Los Angeles,US",
                "London,UK",
                "Tokyo,JP"
            ]
        
        print("Air Quality Monitoring System - DEMO MODE")
        print("="*60)
        
        if self.use_synthetic or not os.getenv('OPENWEATHER_API_KEY'):
            data_source = "synthetic"
        else:
            data_source = "api"
        
        print(f"Analyzing {len(cities)} cities over {days_back} days")
        print("-" * 60)
        
        # Step 1: Data Collection
        all_data = {}
        
        for city in cities:
            print(f"{city}...")
            
            if data_source == "synthetic":
                data = self.data_collector.generate_synthetic_data(city, days_back)
            else:
                data = self.data_collector.collect_historical_data(city, days_back)
            
            if not data.empty:
                all_data[city] = data
                    
        if not all_data:
            print(" No data available for analysis")
            return
        
        # Step 2: Analysis
        analysis_results = {}
        
        for city, data in all_data.items():            
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
                    
        # Step 3: Predictions
        predictions = {}
        
        for city, results in analysis_results.items():            
            try:
                forecast, mae = self.predictor.forecast_aqi(
                    results['data'], 
                    forecast_days=7
                )
                predictions[city] = {
                    'forecast': forecast,
                    'mae': mae
                }
                print(f"    MAE: {mae:.2f}")
            except Exception as e:
                print(f"    Prediction failed: {str(e)}")
                predictions[city] = None
        
        # Step 4: Risk Assessment
        risk_results = {}
        
        for city, results in analysis_results.items():
            print(f"  Assessing {city}...")
            
            risk_metrics = self.risk_assessor.assess_city_risk(
                results['data'], 
                city
            )
            risk_results[city] = risk_metrics
            
            exceedance = risk_metrics['exceedance_factor']
            risk_score = risk_metrics['risk_score']
            risk_level = risk_metrics['risk_classification']['level']
            
            print(f"    Exceedance: {exceedance:.1f}x WHO, Risk: {risk_score:.1f}, Level: {risk_level}")
        
        # Step 5: Generate Summary
        self._print_demo_summary(analysis_results, predictions, risk_results)
        
        # Step 6: Visualizations
        try:
            self.visualizer.create_comprehensive_dashboard(
                analysis_results, predictions, risk_results
            )
            print("    All visualizations saved to 'outputs/plots/'")
        except Exception as e:
            print(f"   Visualization error: {str(e)}")
        
        print("\n" + "="*60)
    
    def _print_demo_summary(self, analysis_results, predictions, risk_results):
        print("\n" + "="*50)
        print(" DEMO RESULTS SUMMARY")
        print("="*50)
        
        # High-risk cities
        high_risk = [(city, risk['exceedance_factor']) 
                    for city, risk in risk_results.items() 
                    if risk['exceedance_factor'] > 2.0]
        
        if high_risk:
            high_risk.sort(key=lambda x: x[1], reverse=True)
            print(f"\n HIGH-RISK CITIES ({len(high_risk)} found):")
            for city, factor in high_risk:
                print(f"   • {city}: {factor:.1f}x WHO limits")
        else:
            print("\n No cities exceeding 2x WHO limits")
        
        # Prediction performance
        mae_values = [p['mae'] for p in predictions.values() if p]
        if mae_values:
            avg_mae = sum(mae_values) / len(mae_values)
            print(f"\n PREDICTION PERFORMANCE:")
            print(f"   • Average MAE: {avg_mae:.2f}")
            print(f"   • Best city: {min(predictions.keys(), key=lambda c: predictions[c]['mae'] if predictions[c] else float('inf'))}")
            print(f"   • Worst city: {max(predictions.keys(), key=lambda c: predictions[c]['mae'] if predictions[c] else 0)}")
        
        # City rankings
        cities_by_risk = sorted(risk_results.keys(), key=lambda c: risk_results[c]['risk_score'], reverse=True)
        
        print(f"\n CITY RANKINGS (by Risk Score):")
        for i, city in enumerate(cities_by_risk[:5], 1):
            risk = risk_results[city]
            print(f"   {i}. {city}: {risk['risk_score']:.1f} ({risk['risk_classification']['level']})")
        
        # Key statistics
        total_population = sum(risk['population_at_risk'] for risk in risk_results.values())
        avg_exceedance = sum(risk['exceedance_factor'] for risk in risk_results.values()) / len(risk_results)
        
        print(f"\n KEY STATISTICS:")
        print(f"   • Cities analyzed: {len(analysis_results)}")
        print(f"   • Total population at risk: {total_population/1e6:.1f}M people")
        print(f"   • Average WHO exceedance: {avg_exceedance:.1f}x")
        print(f"   • Data source: {'Synthetic (Demo)' if self.use_synthetic else 'OpenWeather API'}")

def main():    
    use_synthetic = not os.getenv('OPENWEATHER_API_KEY') or '--demo' in sys.argv
    demo = AirQualityDemo(use_synthetic=use_synthetic)
    
    demo_cities = [
        "Delhi,IN",
        "Beijing,CN",
        "Los Angeles,US",
        "London,UK"
    ]

if __name__ == "__main__":
    main()