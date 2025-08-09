import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AirQualityAnalyzer:    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def seasonal_decomposition(self, data, target_column='aqi', period=24):
        try:
            if len(data) < 2 * period:
                print(f"Warning: Not enough data points for seasonal decomposition. Need at least {2 * period}, got {len(data)}")
                return self._create_empty_decomposition(data, target_column)
            
            data_resampled = data[target_column].resample('H').mean().fillna(method='ffill')
            
            if len(data_resampled.dropna()) < 2 * period:
                return self._create_empty_decomposition(data, target_column)
            
            # Seasonal decomposition
            decomposition = seasonal_decompose(
                data_resampled.dropna(),
                model='additive',
                period=period,
                extrapolate_trend='freq'
            )
            
            result = {
                'original': decomposition.observed,
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'period': period,
                'seasonal_strength': self._calculate_seasonal_strength(decomposition),
                'trend_strength': self._calculate_trend_strength(decomposition)
            }
            
            return result
            
        except Exception as e:
            print(f"Error in seasonal decomposition: {str(e)}")
            return self._create_empty_decomposition(data, target_column)
    
    def _create_empty_decomposition(self, data, target_column):
        return {
            'original': data[target_column],
            'trend': pd.Series(index=data.index, dtype=float),
            'seasonal': pd.Series(index=data.index, dtype=float),
            'residual': pd.Series(index=data.index, dtype=float),
            'period': 24,
            'seasonal_strength': 0.0,
            'trend_strength': 0.0
        }
    
    def _calculate_seasonal_strength(self, decomposition):
        try:
            var_residual = np.var(decomposition.resid.dropna())
            var_seasonal_residual = np.var(decomposition.seasonal.dropna() + decomposition.resid.dropna())
            
            if var_seasonal_residual == 0:
                return 0.0
            
            seasonal_strength = max(0, 1 - var_residual / var_seasonal_residual)
            return seasonal_strength
        except:
            return 0.0
    
    def _calculate_trend_strength(self, decomposition):
        try:
            var_residual = np.var(decomposition.resid.dropna())
            var_trend_residual = np.var(decomposition.trend.dropna() + decomposition.resid.dropna())
            
            if var_trend_residual == 0:
                return 0.0
            
            trend_strength = max(0, 1 - var_residual / var_trend_residual)
            return trend_strength
        except:
            return 0.0
    
    def correlation_analysis(self, data):
        numeric_cols = ['aqi', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
        available_cols = [col for col in numeric_cols if col in data.columns]
        
        if len(available_cols) < 2:
            print("Warning: Not enough numeric columns for correlation analysis")
            return {}
        
        try:
            # Calculate correlation matrix
            corr_data = data[available_cols].dropna()
            
            if len(corr_data) < 10:
                print("Warning: Not enough data points for reliable correlation analysis")
                return {}
            
            # Pearson correlation
            pearson_corr = corr_data.corr(method='pearson')
            
            # Spearman correlation 
            spearman_corr = corr_data.corr(method='spearman')
            
            # Find strongest correlations with AQI
            if 'aqi' in pearson_corr.columns:
                aqi_correlations = pearson_corr['aqi'].drop('aqi').abs().sort_values(ascending=False)
                strongest_aqi_corr = aqi_correlations.head(3)
            else:
                strongest_aqi_corr = pd.Series()
            
            # Statistical significance testing
            significance_results = self._test_correlation_significance(corr_data)
            
            result = {
                'pearson_matrix': pearson_corr,
                'spearman_matrix': spearman_corr,
                'aqi_correlations': strongest_aqi_corr,
                'significance_tests': significance_results,
                'summary': self._summarize_correlations(pearson_corr, strongest_aqi_corr)
            }
            
            return result
            
        except Exception as e:
            print(f"Error in correlation analysis: {str(e)}")
            return {}
    
    def _test_correlation_significance(self, data):
        results = {}
        columns = data.columns
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                try:
                    # Pearson correlation with p-value
                    r, p_value = stats.pearsonr(data[col1].dropna(), data[col2].dropna())
                    results[f"{col1}_vs_{col2}"] = {
                        'correlation': r,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                except:
                    continue
        
        return results
    
    def _summarize_correlations(self, corr_matrix, aqi_corr):
        summary = {
            'total_variables': len(corr_matrix.columns),
            'strongest_correlations': [],
            'aqi_insights': []
        }
        
        corr_values = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                if not np.isnan(corr_val):
                    corr_values.append((abs(corr_val), f"{col1} - {col2}", corr_val))
        
        corr_values.sort(reverse=True)
        summary['strongest_correlations'] = corr_values[:5]
        
        # AQI insights
        if not aqi_corr.empty:
            for pollutant, corr_val in aqi_corr.head(3).items():
                summary['aqi_insights'].append({
                    'pollutant': pollutant,
                    'correlation': corr_val,
                    'interpretation': self._interpret_correlation(corr_val)
                })
        
        return summary
    
    def _interpret_correlation(self, corr_value):
        abs_corr = abs(corr_value)
        if abs_corr >= 0.8:
            return "Very strong"
        elif abs_corr >= 0.6:
            return "Strong"
        elif abs_corr >= 0.4:
            return "Moderate"
        elif abs_corr >= 0.2:
            return "Weak"
        else:
            return "Very weak"
    
    def statistical_summary(self, data):
        try:
            numeric_cols = ['aqi', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
            available_cols = [col for col in numeric_cols if col in data.columns]
            
            if not available_cols:
                return {}
            
            summary = {}
            
            # Basic statistics
            basic_stats = data[available_cols].describe()
            summary['basic_statistics'] = basic_stats
            
            # AQI specific statistics
            if 'aqi' in data.columns:
                aqi_data = data['aqi'].dropna()
                summary.update({
                    'aqi_mean': aqi_data.mean(),
                    'aqi_median': aqi_data.median(),
                    'aqi_std': aqi_data.std(),
                    'aqi_min': aqi_data.min(),
                    'aqi_max': aqi_data.max(),
                    'aqi_range': aqi_data.max() - aqi_data.min()
                })
                
                # AQI distribution
                if 'aqi_category' in data.columns:
                    aqi_distribution = data['aqi_category'].value_counts()
                    summary['aqi_distribution'] = aqi_distribution
                    summary['aqi_distribution_pct'] = (aqi_distribution / len(data) * 100).round(2)
            
            # Temporal patterns
            if 'hour' in data.columns:
                summary['hourly_patterns'] = self._analyze_temporal_patterns(data, 'hour')
            
            if 'day_of_week' in data.columns:
                summary['weekly_patterns'] = self._analyze_temporal_patterns(data, 'day_of_week')
            
            if 'month' in data.columns:
                summary['monthly_patterns'] = self._analyze_temporal_patterns(data, 'month')
            
            # Pollution threshold exceedances
            summary['threshold_analysis'] = self._analyze_threshold_exceedances(data)
            
            # Data quality metrics
            summary['data_quality'] = self._assess_data_quality(data)
            
            return summary
            
        except Exception as e:
            print(f"Error in statistical summary: {str(e)}")
            return {}
    
    def _analyze_temporal_patterns(self, data, time_column):
        try:
            if 'aqi' not in data.columns or time_column not in data.columns:
                return {}
            
            grouped = data.groupby(time_column)['aqi'].agg(['mean', 'std', 'count'])
            
            # Find peak and low periods
            peak_period = grouped['mean'].idxmax()
            low_period = grouped['mean'].idxmin()
            
            return {
                'patterns': grouped,
                'peak_period': peak_period,
                'peak_value': grouped.loc[peak_period, 'mean'],
                'low_period': low_period,
                'low_value': grouped.loc[low_period, 'mean'],
                'variation_coefficient': grouped['mean'].std() / grouped['mean'].mean()
            }
        except:
            return {}
    
    def _analyze_threshold_exceedances(self, data):
        # WHO guidelines (μg/m³)
        thresholds = {
            'pm2_5': 15,
            'pm10': 45,
            'no2': 40,
            'so2': 20,
            'o3': 100
        }
        
        exceedances = {}
        
        for pollutant, threshold in thresholds.items():
            if pollutant in data.columns:
                pollutant_data = data[pollutant].dropna()
                if len(pollutant_data) > 0:
                    exceeding = pollutant_data > threshold
                    exceedances[pollutant] = {
                        'threshold': threshold,
                        'exceedances_count': exceeding.sum(),
                        'exceedances_pct': (exceeding.sum() / len(pollutant_data) * 100).round(2),
                        'max_exceedance': pollutant_data.max() - threshold if pollutant_data.max() > threshold else 0,
                        'avg_when_exceeding': pollutant_data[exceeding].mean() if exceeding.any() else 0
                    }
        
        return exceedances
    
    def _assess_data_quality(self, data):
        total_points = len(data)
        
        quality_metrics = {
            'total_records': total_points,
            'missing_data': {},
            'outliers': {},
            'data_completeness': 0
        }
        
        # Missing data analysis
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            quality_metrics['missing_data'][col] = {
                'count': missing_count,
                'percentage': (missing_count / total_points * 100).round(2)
            }
        
        # Overall completeness
        total_missing = sum(data.isnull().sum())
        total_possible = len(data.columns) * total_points
        quality_metrics['data_completeness'] = ((total_possible - total_missing) / total_possible * 100).round(2)
        
        # Outlier detection using IQR method
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) > 10:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((col_data < lower_bound) | (col_data > upper_bound))
                quality_metrics['outliers'][col] = {
                    'count': outliers.sum(),
                    'percentage': (outliers.sum() / len(col_data) * 100).round(2)
                }
        
        return quality_metrics
    
    def detect_anomalies(self, data, method='iqr', threshold=1.5):
        try:
            numeric_cols = ['aqi', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
            available_cols = [col for col in numeric_cols if col in data.columns]
            
            anomalies = {}
            
            for col in available_cols:
                col_data = data[col].dropna()
                
                if method == 'iqr':
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    anomaly_mask = (col_data < lower_bound) | (col_data > upper_bound)
                    
                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(col_data))
                    anomaly_mask = z_scores > threshold
                
                anomalies[col] = {
                    'anomaly_indices': col_data[anomaly_mask].index.tolist(),
                    'anomaly_values': col_data[anomaly_mask].tolist(),
                    'anomaly_count': anomaly_mask.sum(),
                    'anomaly_percentage': (anomaly_mask.sum() / len(col_data) * 100).round(2)
                }
            
            return anomalies
            
        except Exception as e:
            print(f"Error in anomaly detection: {str(e)}")
            return {}