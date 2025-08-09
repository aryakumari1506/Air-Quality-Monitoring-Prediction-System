import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools

warnings.filterwarnings('ignore')

class ARIMAPredictor:    
    def __init__(self):
        self.best_model = None
        self.best_params = None
        self.model_performance = {}
        
    def forecast_aqi(self, data, forecast_days=7, target_column='aqi'):
        try:
            if target_column not in data.columns:
                raise ValueError(f"Column '{target_column}' not found in data")
            
            ts_data = self._prepare_timeseries(data, target_column)
            
            if len(ts_data) < 20:
                raise ValueError("Not enough data points for reliable ARIMA modeling")
            
            # Find optimal ARIMA parameters
            optimal_params = self._find_optimal_params(ts_data)
            
            # Split data for validation
            split_point = int(len(ts_data) * 0.8)
            train_data = ts_data[:split_point]
            test_data = ts_data[split_point:]
            
            # Train ARIMA model
            model = ARIMA(train_data, order=optimal_params)
            fitted_model = model.fit()
            
            # Calculate MAE on test set
            if len(test_data) > 0:
                test_predictions = fitted_model.forecast(steps=len(test_data))
                mae = mean_absolute_error(test_data, test_predictions)
            else:
                fitted_values = fitted_model.fittedvalues
                mae = mean_absolute_error(train_data[1:], fitted_values[1:])  # Skip first value due to differencing
            
            # Generate forecast
            forecast_steps = forecast_days * 24  # Hourly forecasts
            forecast = fitted_model.forecast(steps=forecast_steps)
            forecast_index = pd.date_range(
                start=ts_data.index[-1] + pd.Timedelta(hours=1),
                periods=forecast_steps,
                freq='H'
            )
            
            # Get confidence intervals
            forecast_ci = fitted_model.get_forecast(steps=forecast_steps).conf_int()
            
            forecast_results = {
                'forecast': pd.Series(forecast, index=forecast_index),
                'confidence_interval': forecast_ci,
                'model_params': optimal_params,
                'model_summary': str(fitted_model.summary()),
                'historical_data': ts_data,
                'fitted_values': fitted_model.fittedvalues,
                'residuals': fitted_model.resid
            }
            
            self.best_model = fitted_model
            self.best_params = optimal_params
            
            return forecast_results, mae
            
        except Exception as e:
            print(f"Error in ARIMA forecasting: {str(e)}")
            # Return simple moving average forecast as fallback
            return self._fallback_forecast(data, target_column, forecast_days), 10.0
    
    def _prepare_timeseries(self, data, target_column):
        ts_data = data[target_column].resample('H').mean()
        ts_data = ts_data.fillna(method='ffill').fillna(method='bfill')
        ts_data = ts_data.dropna()
        
        return ts_data
    
    def _find_optimal_params(self, ts_data, max_p=3, max_d=2, max_q=3):
        try:
            # Check stationarity and determine d
            d = self._determine_differencing_order(ts_data, max_d)
            
            # Grid search for p and q
            best_aic = np.inf
            best_params = (1, d, 1)
            
            p_values = range(0, max_p + 1)
            q_values = range(0, max_q + 1)
            
            for p, q in itertools.product(p_values, q_values):
                try:
                    if p == 0 and q == 0:
                        continue
                    
                    model = ARIMA(ts_data, order=(p, d, q))
                    fitted_model = model.fit()
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_params = (p, d, q)
                        
                except:
                    continue
            
            return best_params
            
        except Exception as e:
            print(f"Error finding optimal parameters: {str(e)}")
            return (1, 1, 1)
    
    def _determine_differencing_order(self, ts_data, max_d=2):
        try:
            # Test original series
            adf_stat, p_value = adfuller(ts_data.dropna())[:2]
            
            if p_value <= 0.05:
                return 0 
            
            # Test first difference
            diff1 = ts_data.diff().dropna()
            if len(diff1) > 0:
                adf_stat, p_value = adfuller(diff1)[:2]
                if p_value <= 0.05:
                    return 1
            
            # Test second difference
            if max_d >= 2:
                diff2 = diff1.diff().dropna()
                if len(diff2) > 0:
                    adf_stat, p_value = adfuller(diff2)[:2]
                    if p_value <= 0.05:
                        return 2
            
            return 1  # Default to first differencing
            
        except:
            return 1  # Default
    
    def _fallback_forecast(self, data, target_column, forecast_days):
        try:
            ts_data = data[target_column].dropna()
            window = min(7 * 24, len(ts_data))  # 7 days or available data
            avg_value = ts_data.tail(window).mean()
            
            # Add some trend and seasonality
            recent_trend = (ts_data.tail(24).mean() - ts_data.head(24).mean()) / len(ts_data)
            
            forecast_steps = forecast_days * 24
            forecast_index = pd.date_range(
                start=ts_data.index[-1] + pd.Timedelta(hours=1),
                periods=forecast_steps,
                freq='H'
            )
            
            # Create forecast with trend and daily seasonality
            forecast_values = []
            for i in range(forecast_steps):
                base_value = avg_value + (recent_trend * i)
                seasonal_component = 0.5 * np.sin(2 * np.pi * (i % 24) / 24)
                noise = np.random.normal(0, 0.1)
                
                forecast_value = max(1, base_value + seasonal_component + noise)
                forecast_values.append(forecast_value)
            
            forecast = pd.Series(forecast_values, index=forecast_index)
            
            # Create confidence interval
            std_dev = ts_data.std()
            lower_ci = forecast - 1.96 * std_dev
            upper_ci = forecast + 1.96 * std_dev
            
            confidence_interval = pd.DataFrame({
                'lower': lower_ci,
                'upper': upper_ci
            }, index=forecast_index)
            
            return {
                'forecast': forecast,
                'confidence_interval': confidence_interval,
                'model_params': 'Moving Average (Fallback)',
                'model_summary': 'Fallback forecast using moving average',
                'historical_data': ts_data,
                'fitted_values': pd.Series(index=ts_data.index, dtype=float),
                'residuals': pd.Series(index=ts_data.index, dtype=float)
            }
            
        except Exception as e:
            print(f"Error in fallback forecast: {str(e)}")
            empty_index = pd.date_range(
                start=pd.Timestamp.now(),
                periods=forecast_days * 24,
                freq='H'
            )
            return {
                'forecast': pd.Series(index=empty_index, dtype=float),
                'confidence_interval': pd.DataFrame(index=empty_index),
                'model_params': 'Error',
                'model_summary': 'Forecast failed',
                'historical_data': pd.Series(dtype=float),
                'fitted_values': pd.Series(dtype=float),
                'residuals': pd.Series(dtype=float)
            }
    
    def evaluate_model_performance(self, actual, predicted):
        try:
            # Align the series
            common_index = actual.index.intersection(predicted.index)
            actual_aligned = actual.loc[common_index]
            predicted_aligned = predicted.loc[common_index]
            
            if len(actual_aligned) == 0:
                return {}
            
            # Calculate metrics
            mae = mean_absolute_error(actual_aligned, predicted_aligned)
            mse = mean_squared_error(actual_aligned, predicted_aligned)
            rmse = np.sqrt(mse)
            
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((actual_aligned - predicted_aligned) / actual_aligned)) * 100
            
            # R-squared
            ss_res = np.sum((actual_aligned - predicted_aligned) ** 2)
            ss_tot = np.sum((actual_aligned - np.mean(actual_aligned)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'r2': r2,
                'n_predictions': len(actual_aligned)
            }
            
        except Exception as e:
            print(f"Error evaluating model performance: {str(e)}")
            return {}
    
    def forecast_multiple_pollutants(self, data, pollutants=None, forecast_days=7):
        if pollutants is None:
            pollutants = ['aqi', 'pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3']
        
        available_pollutants = [p for p in pollutants if p in data.columns]
        
        results = {}
        
        for pollutant in available_pollutants:
            try:
                print(f"  Forecasting {pollutant}...")
                forecast_result, mae = self.forecast_aqi(data, forecast_days, pollutant)
                results[pollutant] = {
                    'forecast': forecast_result,
                    'mae': mae
                }
            except Exception as e:
                print(f"  Failed to forecast {pollutant}: {str(e)}")
                results[pollutant] = None
        
        return results
    
    def detect_forecast_anomalies(self, forecast, confidence_interval, threshold=2.0):
        try:
            anomalies = {}
            
            # Check for values outside confidence intervals
            if 'lower' in confidence_interval.columns and 'upper' in confidence_interval.columns:
                outside_ci = (forecast < confidence_interval['lower']) | (forecast > confidence_interval['upper'])
                anomalies['outside_confidence_interval'] = {
                    'count': outside_ci.sum(),
                    'indices': forecast[outside_ci].index.tolist(),
                    'values': forecast[outside_ci].tolist()
                }
            
            # Check for extreme values (beyond threshold * standard deviation)
            forecast_std = forecast.std()
            forecast_mean = forecast.mean()
            extreme_values = np.abs(forecast - forecast_mean) > (threshold * forecast_std)
            
            anomalies['extreme_values'] = {
                'count': extreme_values.sum(),
                'indices': forecast[extreme_values].index.tolist(),
                'values': forecast[extreme_values].tolist()
            }
            
            return anomalies
            
        except Exception as e:
            print(f"Error detecting forecast anomalies: {str(e)}")
            return {}