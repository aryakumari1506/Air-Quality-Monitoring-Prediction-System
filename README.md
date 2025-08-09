# Air Quality Monitoring & Prediction System

This project is an end-to-end Air Quality Monitoring and Prediction System that collects, analyzes, forecasts, and visualizes air quality data for major global cities. It supports both real-time data from the OpenWeather API and synthetic data generation for demonstration purposes.

## Features

- **Data Collection:** Fetches historical and current air quality data for multiple cities using the OpenWeather API or generates realistic synthetic data.
- **Data Analysis:** Performs statistical summaries, seasonal decomposition, correlation analysis, and data quality assessment.
- **Prediction:** Uses ARIMA time series models to forecast AQI and other pollutant levels.
- **Risk Assessment:** Calculates exceedance factors, risk scores, and health impact assessments based on WHO guidelines and city populations.
- **Visualization:** Generates comprehensive static and interactive dashboards, comparative analyses, and summary reports.
- **Reporting:** Outputs detailed text and visual summary reports highlighting key findings and recommendations.

## Project Structure

```
analyzer.py           # Data analysis and statistics
config.py             # Configuration (API keys, city lists, thresholds)
data_collector.py     # Data collection from API or synthetic generation
demo.py               # Demo script for running the system in demo mode
main.py               # Main entry point for full analysis and reporting
predictor.py          # ARIMA-based forecasting
risk_assessor.py      # Health risk and exceedance assessment
visualizer.py         # Visualization and dashboard generation
requirements.txt      # Python dependencies
outputs/              # Generated plots, dashboards, and reports
```

## Usage

### 1. Setup

- Install dependencies:
  ```sh
  pip install -r requirements.txt
  ```
- (Optional) Set your OpenWeather API key:
  ```sh
  export OPENWEATHER_API_KEY=your_api_key_here
  ```

### 2. Running the System

#### Main Analysis

Run the main analysis and reporting pipeline:
```sh
python main.py
```
- By default, analyzes a set of major cities defined in [`Config.default_cities`](config.py).
- Generates plots in `outputs/` and a summary report in `outputs/summary_report.txt`.

#### Demo Mode

Run the demo script (uses synthetic data if no API key is set):
```sh
python demo.py
```

### 3. Outputs

- **Plots:** Static PNGs for time series, decomposition, risk dashboards, comparative analysis, and forecasts (`outputs/plots/`).
- **Interactive Dashboard:** HTML dashboard with Plotly (`outputs/interactive_dashboard.html`).
- **Summary Report:** Text summary of key findings (`outputs/summary_report.txt`).
- **Visual Summary:** Infographic-style summary (`outputs/summary_report_visual.png`).

## Key Components

- [`AirQualityDataCollector`](data_collector.py): Handles API requests and synthetic data generation.
- [`AirQualityAnalyzer`](analyzer.py): Provides statistical summaries, decomposition, and correlation analysis.
- [`ARIMAPredictor`](predictor.py): Trains ARIMA models and forecasts AQI/pollutants.
- [`RiskAssessor`](risk_assessor.py): Computes risk scores, exceedance factors, and health recommendations.
- [`DataVisualizer`](visualizer.py): Creates all static and interactive visualizations.
- [`Config`](config.py): Centralizes all configuration, thresholds, and city/population data.

## Customization

- **Cities:** Edit `default_cities` in [`Config`](config.py) to change the list of analyzed cities.
- **WHO Limits & Populations:** Adjust `who_limits` and `city_populations` in [`Config`](config.py) or [`RiskAssessor`](risk_assessor.py).
- **Visualization Settings:** Modify `viz_settings` in [`Config`](config.py).

## Requirements

- Python 3.7+
- See [`requirements.txt`](requirements.txt) for all dependencies.

