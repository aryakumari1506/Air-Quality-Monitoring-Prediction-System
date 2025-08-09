import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

class DataVisualizer:    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette("viridis")
        
        self.output_dir = 'outputs/plots'
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.aqi_colors = {
            1: '#00E400',  # Good
            2: '#FFFF00',  # Fair
            3: '#FF7E00',  # Moderate
            4: '#FF0000',  # Poor
            5: '#8F3F97'   # Very Poor
        }
        
        self.risk_colors = {
            'Low': '#00E400',
            'Low-Moderate': '#FFFF00',
            'Moderate': '#FF7E00',
            'High': '#FF0000',
            'Critical': '#8B0000'
        }
    
    def create_comprehensive_dashboard(self, analysis_results, predictions, risk_results):
        # 1. Time Series Overview
        self._create_timeseries_overview(analysis_results)
        
        # 2. Seasonal Decomposition Plots
        self._create_seasonal_decomposition_plots(analysis_results)
        
        # 3. Correlation Heatmaps
        self._create_correlation_heatmaps(analysis_results)
        
        # 4. Prediction Visualizations
        self._create_prediction_plots(analysis_results, predictions)
        
        # 5. Risk Assessment Dashboard
        self._create_risk_dashboard(risk_results)
        
        # 6. Comparative Analysis
        self._create_comparative_analysis(analysis_results, risk_results)
        
        # 7. Interactive Dashboard
        try:
            self._create_interactive_dashboard(analysis_results, predictions, risk_results)
        except ImportError:
            print("    Plotly not available, skipping dashboard")
            
    def _create_timeseries_overview(self, analysis_results):
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('Air Quality Time Series Overview', fontsize=16, fontweight='bold')
        
        # Plot 1: AQI trends for all cities
        ax1 = axes[0, 0]
        for city, results in analysis_results.items():
            data = results['data']
            if 'aqi' in data.columns:
                # Resample to daily for cleaner visualization
                daily_aqi = data['aqi'].resample('D').mean()
                ax1.plot(daily_aqi.index, daily_aqi.values, label=city.split(',')[0], linewidth=2)
        
        ax1.set_title('AQI Trends by City', fontweight='bold')
        ax1.set_ylabel('AQI')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: PM2.5 concentrations
        ax2 = axes[0, 1]
        for city, results in analysis_results.items():
            data = results['data']
            if 'pm2_5' in data.columns:
                daily_pm25 = data['pm2_5'].resample('D').mean()
                ax2.plot(daily_pm25.index, daily_pm25.values, label=city.split(',')[0], linewidth=2)
        
        ax2.set_title('PM2.5 Concentrations by City', fontweight='bold')
        ax2.set_ylabel('PM2.5 (μg/m³)')
        ax2.axhline(y=15, color='red', linestyle='--', alpha=0.7, label='WHO Limit')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: AQI distribution
        ax3 = axes[1, 0]
        aqi_data = []
        city_labels = []
        
        for city, results in analysis_results.items():
            data = results['data']
            if 'aqi' in data.columns:
                aqi_data.extend(data['aqi'].dropna().tolist())
                city_labels.extend([city.split(',')[0]] * len(data['aqi'].dropna()))
        
        if aqi_data:
            df_box = pd.DataFrame({'AQI': aqi_data, 'City': city_labels})
            sns.boxplot(data=df_box, x='City', y='AQI', ax=ax3)
            ax3.set_title('AQI Distribution by City', fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Hourly patterns
        ax4 = axes[1, 1]
        for city, results in analysis_results.items():
            data = results['data']
            if 'aqi' in data.columns and 'hour' in data.columns:
                hourly_pattern = data.groupby('hour')['aqi'].mean()
                ax4.plot(hourly_pattern.index, hourly_pattern.values, label=city.split(',')[0], linewidth=2, marker='o', markersize=4)
        
        ax4.set_title('Average AQI by Hour of Day', fontweight='bold')
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel('Average AQI')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(range(0, 24, 3))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/timeseries_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_seasonal_decomposition_plots(self, analysis_results):
        for city, results in analysis_results.items():
            decomposition = results.get('decomposition', {})
            
            if not decomposition or decomposition.get('original') is None:
                continue
            
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))
            fig.suptitle(f'Seasonal Decomposition - {city}', fontsize=16, fontweight='bold')
            
            # Original series
            axes[0].plot(decomposition['original'].index, decomposition['original'].values)
            axes[0].set_title('Original AQI')
            axes[0].set_ylabel('AQI')
            axes[0].grid(True, alpha=0.3)
            
            # Trend
            if not decomposition['trend'].empty:
                axes[1].plot(decomposition['trend'].index, decomposition['trend'].values, color='orange')
                axes[1].set_title(f'Trend (Strength: {decomposition.get("trend_strength", 0):.3f})')
                axes[1].set_ylabel('AQI')
                axes[1].grid(True, alpha=0.3)
            
            # Seasonal
            if not decomposition['seasonal'].empty:
                axes[2].plot(decomposition['seasonal'].index, decomposition['seasonal'].values, color='green')
                axes[2].set_title(f'Seasonal (Strength: {decomposition.get("seasonal_strength", 0):.3f})')
                axes[2].set_ylabel('AQI')
                axes[2].grid(True, alpha=0.3)
            
            # Residual
            if not decomposition['residual'].empty:
                axes[3].plot(decomposition['residual'].index, decomposition['residual'].values, color='red')
                axes[3].set_title('Residual')
                axes[3].set_ylabel('AQI')
                axes[3].set_xlabel('Date')
                axes[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            safe_city_name = city.replace(',', '_').replace(' ', '_')
            plt.savefig(f'{self.output_dir}/seasonal_decomposition_{safe_city_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_correlation_heatmaps(self, analysis_results):
        n_cities = len(analysis_results)
        if n_cities == 0:
            return
        
        cols = min(3, n_cities)
        rows = (n_cities + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if n_cities == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle('Pollution Correlation Heatmaps', fontsize=16, fontweight='bold')
        
        for idx, (city, results) in enumerate(analysis_results.items()):
            correlations = results.get('correlations', {})
            
            if 'pearson_matrix' in correlations and not correlations['pearson_matrix'].empty:
                ax = axes[idx] if n_cities > 1 else axes[0]
                
                # Create heatmap
                sns.heatmap(correlations['pearson_matrix'], annot=True, fmt='.2f', cmap='coolwarm', center=0,square=True, ax=ax, cbar_kws={'shrink': 0.8})
                
                ax.set_title(f'{city}', fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
                ax.tick_params(axis='y', rotation=0)
            
        for idx in range(n_cities, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/correlation_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_prediction_plots(self, analysis_results, predictions):
        for city, pred_result in predictions.items():
            if not pred_result:
                continue
            
            forecast_data = pred_result['forecast']
            mae = pred_result['mae']
            
            if 'forecast' not in forecast_data:
                continue
            
            forecast = forecast_data['forecast']
            historical = forecast_data.get('historical_data', pd.Series())
            confidence_interval = forecast_data.get('confidence_interval', pd.DataFrame())
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            fig.suptitle(f'AQI Forecast - {city} (MAE: {mae:.2f})', fontsize=16, fontweight='bold')
            
            # Plot 1: Historical + Forecast
            if not historical.empty:
                recent_historical = historical.tail(7*24)  # Last 7 days (hourly data)
                ax1.plot(recent_historical.index, recent_historical.values, 
                        label='Historical', color='blue', linewidth=2)
            
            if not forecast.empty:
                ax1.plot(forecast.index, forecast.values, label='Forecast', color='red', linewidth=2, linestyle='--')
                
                if not confidence_interval.empty and 'lower' in confidence_interval.columns:
                    ax1.fill_between(forecast.index, confidence_interval['lower'], confidence_interval['upper'], alpha=0.3, color='red', label='Confidence Interval')
            
            ax1.set_title('Historical Data and Forecast')
            ax1.set_ylabel('AQI')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add AQI threshold lines
            ax1.axhline(y=50, color='orange', linestyle=':', alpha=0.7, label='Moderate')
            ax1.axhline(y=100, color='red', linestyle=':', alpha=0.7, label='Unhealthy for Sensitive Groups')
            
            # Plot 2: Residuals analysis
            residuals = forecast_data.get('residuals', pd.Series())
            if not residuals.empty and len(residuals.dropna()) > 0:
                ax2.plot(residuals.index, residuals.values, color='green', alpha=0.7)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                ax2.set_title('Model Residuals')
                ax2.set_ylabel('Residuals')
                ax2.set_xlabel('Date')
                ax2.grid(True, alpha=0.3)
                
                # Add residual statistics
                residual_stats = f'Mean: {residuals.mean():.2f}, Std: {residuals.std():.2f}'
                ax2.text(0.02, 0.98, residual_stats, transform=ax2.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                ax2.text(0.5, 0.5, 'No residuals available', ha='center', va='center', transform=ax2.transAxes, fontsize=14)
                ax2.set_title('Model Residuals - Not Available')
            
            plt.tight_layout()
            safe_city_name = city.replace(',', '_').replace(' ', '_')
            plt.savefig(f'{self.output_dir}/forecast_{safe_city_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_risk_dashboard(self, risk_results):
        if not risk_results:
            return
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Risk Score Comparison (Top left)
        ax1 = fig.add_subplot(gs[0, :2])
        cities = list(risk_results.keys())
        risk_scores = [risk_results[city]['risk_score'] for city in cities]
        exceedance_factors = [risk_results[city]['exceedance_factor'] for city in cities]
        
        bars = ax1.bar(range(len(cities)), risk_scores, color=[self.risk_colors.get(risk_results[city]['risk_classification']['level'], '#CCCCCC') for city in cities])
        ax1.set_title('Population-Weighted Risk Scores', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Risk Score')
        ax1.set_xticks(range(len(cities)))
        ax1.set_xticklabels([city.split(',')[0] for city in cities], rotation=45)
        
        for i, (bar, score) in enumerate(zip(bars, risk_scores)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Exceedance Factors (Top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        bars2 = ax2.bar(range(len(cities)), exceedance_factors, color='coral')
        ax2.set_title('Average Exceedance Factors (WHO Limits)', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Exceedance Factor')
        ax2.set_xticks(range(len(cities)))
        ax2.set_xticklabels([city.split(',')[0] for city in cities], rotation=45)
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='WHO Limit')
        ax2.axhline(y=2, color='darkred', linestyle='--', alpha=0.7, label='2x WHO Limit')
        ax2.legend()
        
        # Add value labels
        for bar, factor in zip(bars2, exceedance_factors):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{factor:.1f}x', ha='center', va='bottom', fontweight='bold')
        
        # 3. Days Exceeding WHO Limits (Middle left)
        ax3 = fig.add_subplot(gs[1, :2])
        days_exceeding = [risk_results[city]['days_exceeding_who'] for city in cities]
        bars3 = ax3.bar(range(len(cities)), days_exceeding, color='lightcoral')
        ax3.set_title('Days Exceeding WHO Limits', fontweight='bold', fontsize=14)
        ax3.set_ylabel('Number of Days')
        ax3.set_xticks(range(len(cities)))
        ax3.set_xticklabels([city.split(',')[0] for city in cities], rotation=45)
        
        # 4. Population at Risk (Middle right)
        ax4 = fig.add_subplot(gs[1, 2:])
        population_at_risk = [risk_results[city]['population_at_risk'] / 1e6 for city in cities]  # In millions
        bars4 = ax4.bar(range(len(cities)), population_at_risk, color='lightblue')
        ax4.set_title('Population at Risk (Millions)', fontweight='bold', fontsize=14)
        ax4.set_ylabel('Population (Millions)')
        ax4.set_xticks(range(len(cities)))
        ax4.set_xticklabels([city.split(',')[0] for city in cities], rotation=45)
        
        # 5. Pollutant-specific Exceedances (Bottom - spanning 2x2)
        ax5 = fig.add_subplot(gs[2:, :])
        
        pollutants = ['pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3']
        exceedance_matrix = []
        
        for city in cities:
            city_exceedances = []
            for pollutant in pollutants:
                factor = risk_results[city]['exceedance_factors_by_pollutant'].get(pollutant, 0)
                city_exceedances.append(factor)
            exceedance_matrix.append(city_exceedances)
        
        exceedance_df = pd.DataFrame(exceedance_matrix, columns=[p.upper().replace('_', '.') for p in pollutants],index=[city.split(',')[0] for city in cities])
        
        sns.heatmap(exceedance_df, annot=True, fmt='.1f', cmap='Reds', center=1, ax=ax5, cbar_kws={'label': 'Exceedance Factor'})
        ax5.set_title('Pollutant-Specific Exceedance Factors', fontweight='bold', fontsize=14)
        ax5.set_xlabel('Pollutants')
        ax5.set_ylabel('Cities')
        
        plt.suptitle('Air Quality Risk Assessment Dashboard', fontsize=20, fontweight='bold', y=0.98)
        plt.savefig(f'{self.output_dir}/risk_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_comparative_analysis(self, analysis_results, risk_results):
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Comparative Analysis Across Cities', fontsize=16, fontweight='bold')
        cities = list(analysis_results.keys())
        
        # 1. AQI Statistics Comparison
        ax1 = axes[0, 0]
        aqi_means = []
        aqi_maxs = []
        aqi_stds = []
        
        for city in cities:
            stats = analysis_results[city].get('statistics', {})
            aqi_means.append(stats.get('aqi_mean', 0))
            aqi_maxs.append(stats.get('aqi_max', 0))
            aqi_stds.append(stats.get('aqi_std', 0))
        
        x = np.arange(len(cities))
        width = 0.25
        
        ax1.bar(x - width, aqi_means, width, label='Mean', alpha=0.8)
        ax1.bar(x, aqi_maxs, width, label='Max', alpha=0.8)
        ax1.bar(x + width, aqi_stds, width, label='Std Dev', alpha=0.8)
        
        ax1.set_title('AQI Statistics by City')
        ax1.set_ylabel('AQI Value')
        ax1.set_xticks(x)
        ax1.set_xticklabels([city.split(',')[0] for city in cities], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. PM2.5 vs WHO Limit
        ax2 = axes[0, 1]
        pm25_means = []
        
        for city in cities:
            data = analysis_results[city]['data']
            if 'pm2_5' in data.columns:
                pm25_means.append(data['pm2_5'].mean())
            else:
                pm25_means.append(0)
        
        bars = ax2.bar(range(len(cities)), pm25_means, color='brown', alpha=0.7)
        ax2.axhline(y=15, color='red', linestyle='--', linewidth=2, label='WHO Limit (15 μg/m³)')
        ax2.set_title('PM2.5 Concentrations vs WHO Limit')
        ax2.set_ylabel('PM2.5 (μg/m³)')
        ax2.set_xticks(range(len(cities)))
        ax2.set_xticklabels([city.split(',')[0] for city in cities], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Risk Classification Distribution
        ax3 = axes[0, 2]
        risk_levels = [risk_results[city]['risk_classification']['level'] for city in cities]
        risk_counts = pd.Series(risk_levels).value_counts()
        
        colors = [self.risk_colors.get(level, '#CCCCCC') for level in risk_counts.index]
        wedges, texts, autotexts = ax3.pie(risk_counts.values, labels=risk_counts.index, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Risk Level Distribution')
        
        # 4. Seasonal Patterns Comparison
        ax4 = axes[1, 0]
        for city in cities:
            data = analysis_results[city]['data']
            if 'aqi' in data.columns and 'month' in data.columns:
                monthly_pattern = data.groupby('month')['aqi'].mean()
                ax4.plot(monthly_pattern.index, monthly_pattern.values, label=city.split(',')[0], marker='o', linewidth=2)
        
        ax4.set_title('Seasonal AQI Patterns')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Average AQI')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(range(1, 13))
        
        # 5. Correlation Strength Comparison
        ax5 = axes[1, 1]
        strongest_corrs = []
        
        for city in cities:
            correlations = analysis_results[city].get('correlations', {})
            if 'aqi_correlations' in correlations and not correlations['aqi_correlations'].empty:
                strongest_corr = correlations['aqi_correlations'].iloc[0]
                strongest_corrs.append(abs(strongest_corr))
            else:
                strongest_corrs.append(0)
        
        bars = ax5.bar(range(len(cities)), strongest_corrs, color='purple', alpha=0.7)
        ax5.set_title('Strongest AQI Correlations')
        ax5.set_ylabel('Absolute Correlation')
        ax5.set_xticks(range(len(cities)))
        ax5.set_xticklabels([city.split(',')[0] for city in cities], rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # 6. Data Quality Comparison
        ax6 = axes[1, 2]
        completeness_scores = []
        
        for city in cities:
            stats = analysis_results[city].get('statistics', {})
            quality = stats.get('data_quality', {})
            completeness = quality.get('data_completeness', 0)
            completeness_scores.append(completeness)
        
        bars = ax6.bar(range(len(cities)), completeness_scores, color='green', alpha=0.7)
        ax6.set_title('Data Completeness')
        ax6.set_ylabel('Completeness (%)')
        ax6.set_xticks(range(len(cities)))
        ax6.set_xticklabels([city.split(',')[0] for city in cities], rotation=45)
        ax6.set_ylim(0, 100)
        ax6.grid(True, alpha=0.3)
        
        # Add percentage labels
        for bar, score in zip(bars, completeness_scores):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{score:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/comparative_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_interactive_dashboard(self, analysis_results, predictions, risk_results):
        try:
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=['AQI Time Series', 'Risk Scores', 'PM2.5 vs WHO Limit', 'Prediction Results', 'Exceedance Factors', 'Correlation Matrix'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            cities = list(analysis_results.keys())
            colors = px.colors.qualitative.Set1[:len(cities)]
            
            # 1. AQI Time Series
            for i, city in enumerate(cities):
                data = analysis_results[city]['data']
                if 'aqi' in data.columns:
                    daily_aqi = data['aqi'].resample('D').mean()
                    fig.add_trace(
                        go.Scatter(x=daily_aqi.index, y=daily_aqi.values,mode='lines', name=f'{city.split(",")[0]} AQI',
                                 line=dict(color=colors[i])),
                        row=1, col=1
                    )
            
            # 2. Risk Scores
            risk_scores = [risk_results[city]['risk_score'] for city in cities]
            city_names = [city.split(',')[0] for city in cities]
            
            fig.add_trace(
                go.Bar(x=city_names, y=risk_scores, name='Risk Scores',marker_color='rgba(255, 99, 132, 0.8)'),row=1, col=2
            )
            
            # 3. PM2.5 vs WHO Limit
            pm25_means = []
            for city in cities:
                data = analysis_results[city]['data']
                pm25_means.append(data['pm2_5'].mean() if 'pm2_5' in data.columns else 0)
            
            fig.add_trace(
                go.Bar(x=city_names, y=pm25_means, name='PM2.5',marker_color='rgba(139, 69, 19, 0.8)'),row=2, col=1
            )
            
            # Add WHO limit line
            fig.add_hline(y=15, line_dash="dash", line_color="red", annotation_text="WHO Limit", row=2, col=1)
            
            # 4. Prediction Results (show MAE for each city)
            mae_values = []
            for city in cities:
                pred_result = predictions.get(city)
                mae_values.append(pred_result['mae'] if pred_result else 0)
            
            fig.add_trace(
                go.Bar(x=city_names, y=mae_values, name='Prediction MAE',marker_color='rgba(75, 192, 192, 0.8)'),row=2, col=2
            )
            
            # 5. Exceedance Factors
            exceedance_factors = [risk_results[city]['exceedance_factor'] for city in cities]
            
            fig.add_trace(
                go.Bar(x=city_names, y=exceedance_factors, name='Exceedance Factors',marker_color='rgba(255, 159, 64, 0.8)'),row=3, col=1
            )
            
            fig.add_hline(y=1, line_dash="dash", line_color="orange", annotation_text="WHO Limit", row=3, col=1)
            fig.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="2x WHO Limit", row=3, col=1)
            
            # 6. Correlation Matrix 
            if cities:
                first_city = cities[0]
                correlations = analysis_results[first_city].get('correlations', {})
                if 'pearson_matrix' in correlations:
                    corr_matrix = correlations['pearson_matrix']
                    
                    fig.add_trace(
                        go.Heatmap(z=corr_matrix.values,x=corr_matrix.columns,y=corr_matrix.index,colorscale='RdBu',zmid=0,name=f'Correlations ({first_city.split(",")[0]})'),row=3, col=2
                    )
            
            # Update layout
            fig.update_layout(
                height=1200,
                title_text="Interactive Air Quality Dashboard",
                title_x=0.5,
                showlegend=True
            )
            
            fig.write_html(f'{self.output_dir}/interactive_dashboard.html')
            
        except Exception as e:
            print(f"Error creating dashboard: {str(e)}")
    
    def create_summary_report_visual(self, analysis_results, predictions, risk_results):
        fig = plt.figure(figsize=(16, 20))
        
        fig.text(0.5, 0.98, 'AIR QUALITY MONITORING & PREDICTION SYSTEM', ha='center', va='top', fontsize=20, fontweight='bold')
        fig.text(0.5, 0.96, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ha='center', va='top', fontsize=12)
        
        # Key findings section
        gs = fig.add_gridspec(6, 2, hspace=0.4, wspace=0.3, left=0.1, right=0.9, top=0.93, bottom=0.05)
        
        # High-risk cities
        ax1 = fig.add_subplot(gs[0, :])
        high_risk_cities = [(city, risk_results[city]['exceedance_factor']) for city in risk_results.keys() if risk_results[city]['exceedance_factor'] > 2.0]
        high_risk_cities.sort(key=lambda x: x[1], reverse=True)
        
        ax1.text(0.02, 0.8, 'HIGH-RISK CITIES (Exceeding WHO Limits by 2x+):', fontweight='bold', fontsize=14, transform=ax1.transAxes)
        
        if high_risk_cities:
            avg_exceedance = np.mean([factor for _, factor in high_risk_cities])
            report_text = f"Found {len(high_risk_cities)} high-risk cities\n"
            report_text += f"Average exceedance factor: {avg_exceedance:.1f}x WHO limits\n\n"
            
            for i, (city, factor) in enumerate(high_risk_cities[:5]):
                report_text += f"• {city}: {factor:.2f}x WHO limits\n"
            
        else:
            report_text = "No cities found exceeding WHO limits by 2x"
        
        ax1.text(0.02, 0.6, report_text, fontsize=11, transform=ax1.transAxes,verticalalignment='top')
        ax1.axis('off')
        
        # Prediction accuracy
        ax2 = fig.add_subplot(gs[1, :])
        mae_values = [pred['mae'] for pred in predictions.values() if pred]
        
        ax2.text(0.02, 0.8, 'PREDICTION ACCURACY:', fontweight='bold', fontsize=14, transform=ax2.transAxes)
        
        if mae_values:
            avg_mae = np.mean(mae_values)
            accuracy_text = f"Average MAE across all cities: {avg_mae:.2f}\n\n"
            
            for city, pred in predictions.items():
                if pred:
                    accuracy_text += f"• {city}: MAE = {pred['mae']:.2f}\n"
        else:
            accuracy_text = "No prediction results available"
        
        ax2.text(0.02, 0.6, accuracy_text, fontsize=11, transform=ax2.transAxes,verticalalignment='top')
        ax2.axis('off')
        
        cities = list(analysis_results.keys())
        
        # Risk scores bar chart
        ax3 = fig.add_subplot(gs[2, 0])
        risk_scores = [risk_results[city]['risk_score'] for city in cities]
        bars = ax3.bar(range(len(cities)), risk_scores, color='coral')
        ax3.set_title('Risk Scores by City', fontweight='bold')
        ax3.set_xticks(range(len(cities)))
        ax3.set_xticklabels([city.split(',')[0] for city in cities], rotation=45)
        ax3.set_ylabel('Risk Score')
        
        # Exceedance factors
        ax4 = fig.add_subplot(gs[2, 1])
        exceedance_factors = [risk_results[city]['exceedance_factor'] for city in cities]
        bars = ax4.bar(range(len(cities)), exceedance_factors, color='lightcoral')
        ax4.axhline(y=2, color='red', linestyle='--', alpha=0.7)
        ax4.set_title('Exceedance Factors', fontweight='bold')
        ax4.set_xticks(range(len(cities)))
        ax4.set_xticklabels([city.split(',')[0] for city in cities], rotation=45)
        ax4.set_ylabel('Exceedance Factor')
        
        # AQI trends
        ax5 = fig.add_subplot(gs[3, :])
        for city in cities:
            data = analysis_results[city]['data']
            if 'aqi' in data.columns:
                daily_aqi = data['aqi'].resample('D').mean()
                ax5.plot(daily_aqi.index, daily_aqi.values, label=city.split(',')[0], linewidth=2)
        
        ax5.set_title('AQI Trends Over Time', fontweight='bold')
        ax5.set_ylabel('AQI')
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(gs[4, :])
        ax6.text(0.02, 0.9, 'MODEL PERFORMANCE SUMMARY:', fontweight='bold', fontsize=14, transform=ax6.transAxes)
        
        performance_text = f"• Total cities analyzed: {len(cities)}\n"
        performance_text += f"• Average prediction MAE: {np.mean(mae_values):.2f}\n" if mae_values else "• No predictions available\n"
        performance_text += f"• Cities exceeding 2x WHO limits: {len([c for c in cities if risk_results[c]['exceedance_factor'] > 2.0])}\n"
        performance_text += f"• Highest risk city: {max(cities, key=lambda c: risk_results[c]['risk_score'])}\n"
        
        ax6.text(0.02, 0.7, performance_text, fontsize=12, transform=ax6.transAxes, verticalalignment='top')
        ax6.axis('off')
        
        # Recommendations
        ax7 = fig.add_subplot(gs[5, :])
        ax7.text(0.02, 0.9, '💡 KEY RECOMMENDATIONS:', fontweight='bold', fontsize=14, transform=ax7.transAxes)
        
        recommendations = [
            "• Implement immediate air quality interventions in high-risk cities",
            "• Enhance monitoring networks in cities exceeding WHO limits",
            "• Develop targeted health advisories for vulnerable populations",
            "• Consider seasonal patterns in pollution control measures",
            "• Improve prediction model accuracy through enhanced data collection"
        ]
        
        rec_text = '\n'.join(recommendations)
        ax7.text(0.02, 0.75, rec_text, fontsize=11, transform=ax7.transAxes,
                verticalalignment='top')
        ax7.axis('off')
        
        plt.savefig(f'{self.output_dir}/summary_report_visual.png', dpi=300, bbox_inches='tight')
        plt.close()