import pandas as pd
import numpy as np
from datetime import datetime

class RiskAssessor:    
    def __init__(self):
        # WHO Air Quality Guidelines (μg/m³)
        self.who_limits = {
            'pm2_5': 15,    # PM2.5 annual mean
            'pm10': 45,     # PM10 annual mean
            'no2': 40,      # NO2 annual mean
            'so2': 20,      # SO2 daily mean
            'co': 30000,    # CO 8-hour mean (converted to μg/m³)
            'o3': 100       # O3 8-hour daily maximum mean
        }
        
        # Population data for major cities (millions)
        self.city_populations = {
            'Delhi,IN': 30.29,
            'Beijing,CN': 21.54,
            'Los Angeles,US': 3.97,
            'London,UK': 9.54,
            'Tokyo,JP': 13.96,
            'Mumbai,IN': 20.41,
            'Mexico City,MX': 9.21,
            'São Paulo,BR': 12.33,
            'Cairo,EG': 10.23,
            'Shanghai,CN': 27.05,
            'Dhaka,BD': 9.84,
            'Lagos,NG': 15.39,
            'Karachi,PK': 16.09,
            'Istanbul,TR': 15.46,
            'Bangkok,TH': 5.78
        }
        
        # Health impact weights for different pollutants
        self.health_weights = {
            'pm2_5': 1.0,   # Highest impact
            'pm10': 0.8,
            'no2': 0.7,
            'so2': 0.6,
            'co': 0.5,
            'o3': 0.7
        }
    
    def assess_city_risk(self, data, city_name):
        try:
            # Calculate exceedance factors
            exceedance_factors = self._calculate_exceedance_factors(data)
            
            # Calculate population-weighted risk score
            risk_score = self._calculate_population_weighted_risk(data, city_name, exceedance_factors)
            
            # Calculate daily exceedance factors
            daily_exceedances = self._calculate_daily_exceedances(data)
            
            # Assess health impact levels
            health_impact = self._assess_health_impact(data, exceedance_factors)
            
            # Calculate temporal risk patterns
            temporal_risk = self._analyze_temporal_risk_patterns(data)
            
            # Overall risk classification
            risk_classification = self._classify_risk_level(risk_score, exceedance_factors)
            
            # Count days exceeding WHO limits
            days_exceeding_who = self._count_days_exceeding_who(data)
            
            # Calculate average exceedance factor
            avg_exceedance_factor = np.mean([ef for ef in exceedance_factors.values() if ef > 0])
            
            return {
                'city': city_name,
                'exceedance_factor': avg_exceedance_factor,
                'risk_score': risk_score,
                'exceedance_factors_by_pollutant': exceedance_factors,
                'daily_exceedances': daily_exceedances,
                'health_impact': health_impact,
                'temporal_risk': temporal_risk,
                'risk_classification': risk_classification,
                'days_exceeding_who': days_exceeding_who,
                'population_at_risk': self.city_populations.get(city_name, 0) * 1e6,  # Convert to actual population
                'assessment_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error assessing risk for {city_name}: {str(e)}")
            return self._create_empty_risk_assessment(city_name)
    
    def _calculate_exceedance_factors(self, data):
        exceedance_factors = {}
        
        for pollutant, who_limit in self.who_limits.items():
            if pollutant in data.columns:
                pollutant_data = data[pollutant].dropna()
                
                if len(pollutant_data) > 0:
                    # Calculate average concentration
                    avg_concentration = pollutant_data.mean()
                    
                    # Calculate exceedance factor
                    exceedance_factor = avg_concentration / who_limit
                    exceedance_factors[pollutant] = exceedance_factor
                else:
                    exceedance_factors[pollutant] = 0.0
            else:
                exceedance_factors[pollutant] = 0.0
        
        return exceedance_factors
    
    def _calculate_population_weighted_risk(self, data, city_name, exceedance_factors):
        try:
            population = self.city_populations.get(city_name, 1.0)  # Default to 1M if not found
            
            # Base risk score from exceedance factors and health weights
            weighted_risk = 0.0
            total_weight = 0.0
            
            for pollutant, exceedance_factor in exceedance_factors.items():
                if pollutant in self.health_weights:
                    weight = self.health_weights[pollutant]
                    weighted_risk += exceedance_factor * weight
                    total_weight += weight
            
            if total_weight > 0:
                base_risk = weighted_risk / total_weight
            else:
                base_risk = 0.0
            
            # Apply population weighting 
            population_factor = np.log10(population + 1) / np.log10(30)  # Normalized to Delhi's population
            
            # Final risk score
            risk_score = base_risk * population_factor
            
            return min(10.0, max(0.0, risk_score))  # Clamp between 0 and 10
            
        except Exception as e:
            print(f"Error calculating population-weighted risk: {str(e)}")
            return 0.0
    
    def _calculate_daily_exceedances(self, data):
        try:
            # Resample to daily averages
            daily_data = data.resample('D').mean()
            
            daily_exceedances = {}
            
            for pollutant, who_limit in self.who_limits.items():
                if pollutant in daily_data.columns:
                    daily_pollutant = daily_data[pollutant].dropna()
                    
                    if len(daily_pollutant) > 0:
                        # Calculate daily exceedance factors
                        daily_factors = daily_pollutant / who_limit
                        
                        daily_exceedances[pollutant] = {
                            'mean_daily_factor': daily_factors.mean(),
                            'max_daily_factor': daily_factors.max(),
                            'days_exceeding': (daily_factors > 1.0).sum(),
                            'days_exceeding_pct': (daily_factors > 1.0).sum() / len(daily_factors) * 100,
                            'severe_days': (daily_factors > 2.0).sum(),  # Days exceeding 2x WHO limit
                            'extreme_days': (daily_factors > 3.0).sum()  # Days exceeding 3x WHO limit
                        }
                    else:
                        daily_exceedances[pollutant] = self._create_empty_daily_exceedance()
                else:
                    daily_exceedances[pollutant] = self._create_empty_daily_exceedance()
            
            return daily_exceedances
            
        except Exception as e:
            print(f"Error calculating daily exceedances: {str(e)}")
            return {}
    
    def _create_empty_daily_exceedance(self):
        return {
            'mean_daily_factor': 0.0,
            'max_daily_factor': 0.0,
            'days_exceeding': 0,
            'days_exceeding_pct': 0.0,
            'severe_days': 0,
            'extreme_days': 0
        }
    
    def _assess_health_impact(self, data, exceedance_factors):
        health_impact = {
            'overall_level': 'Low',
            'primary_concerns': [],
            'vulnerable_groups_risk': 'Low',
            'recommended_actions': []
        }
        
        try:
            # Determine overall health impact level
            max_exceedance = max(exceedance_factors.values()) if exceedance_factors else 0
            
            if max_exceedance >= 3.0:
                health_impact['overall_level'] = 'Severe'
                health_impact['vulnerable_groups_risk'] = 'Very High'
            elif max_exceedance >= 2.0:
                health_impact['overall_level'] = 'High'
                health_impact['vulnerable_groups_risk'] = 'High'
            elif max_exceedance >= 1.5:
                health_impact['overall_level'] = 'Moderate'
                health_impact['vulnerable_groups_risk'] = 'Moderate'
            elif max_exceedance >= 1.0:
                health_impact['overall_level'] = 'Low-Moderate'
                health_impact['vulnerable_groups_risk'] = 'Low-Moderate'
            
            # Identify primary concerns
            for pollutant, factor in exceedance_factors.items():
                if factor >= 1.5:
                    health_impact['primary_concerns'].append({
                        'pollutant': pollutant,
                        'exceedance_factor': factor,
                        'health_effects': self._get_health_effects(pollutant)
                    })
            
            # Generate recommendations
            health_impact['recommended_actions'] = self._generate_health_recommendations(
                health_impact['overall_level'],
                health_impact['primary_concerns']
            )
            
        except Exception as e:
            print(f"Error assessing health impact: {str(e)}")
        
        return health_impact
    
    def _get_health_effects(self, pollutant):
        health_effects = {
            'pm2_5': 'Respiratory and cardiovascular disease, lung cancer, premature death',
            'pm10': 'Respiratory irritation, reduced lung function, cardiovascular effects',
            'no2': 'Respiratory inflammation, reduced immune response, asthma exacerbation',
            'so2': 'Respiratory irritation, bronchoconstriction, cardiovascular effects',
            'co': 'Reduced oxygen delivery, cardiovascular stress, neurological effects',
            'o3': 'Respiratory inflammation, chest pain, reduced lung function'
        }
        
        return health_effects.get(pollutant, 'Various respiratory and cardiovascular effects')
    
    def _generate_health_recommendations(self, risk_level, primary_concerns):
        recommendations = []
        
        if risk_level in ['Severe', 'High']:
            recommendations.extend([
                'Limit outdoor activities, especially for sensitive groups',
                'Use air purifiers indoors with HEPA filters',
                'Wear N95 masks when going outside',
                'Keep windows closed and use air conditioning if available',
                'Consult healthcare providers if experiencing symptoms'
            ])
        elif risk_level in ['Moderate', 'Low-Moderate']:
            recommendations.extend([
                'Reduce prolonged outdoor exertion',
                'Consider indoor air filtration',
                'Monitor air quality forecasts',
                'Limit outdoor activities during peak pollution hours'
            ])
        else:
            recommendations.extend([
                'Maintain awareness of air quality conditions',
                'Follow general health guidelines for outdoor activities'
            ])
        
        # Add pollutant-specific recommendations
        for concern in primary_concerns:
            pollutant = concern['pollutant']
            if pollutant in ['pm2_5', 'pm10']:
                recommendations.append('Use particulate matter masks during high PM days')
            elif pollutant == 'o3':
                recommendations.append('Avoid outdoor activities during afternoon hours when ozone peaks')
            elif pollutant in ['no2', 'so2']:
                recommendations.append('Avoid high-traffic areas and industrial zones')
        
        return list(set(recommendations))  
    
    def _analyze_temporal_risk_patterns(self, data):
        temporal_risk = {}
        
        try:
            if 'aqi' in data.columns:
                # Hourly patterns
                if 'hour' in data.columns:
                    hourly_risk = data.groupby('hour')['aqi'].mean()
                    temporal_risk['hourly'] = {
                        'peak_hour': hourly_risk.idxmax(),
                        'peak_aqi': hourly_risk.max(),
                        'low_hour': hourly_risk.idxmin(),
                        'low_aqi': hourly_risk.min()
                    }
                
                # Weekly patterns
                if 'day_of_week' in data.columns:
                    weekly_risk = data.groupby('day_of_week')['aqi'].mean()
                    temporal_risk['weekly'] = {
                        'peak_day': weekly_risk.idxmax(),
                        'peak_aqi': weekly_risk.max(),
                        'low_day': weekly_risk.idxmin(),
                        'low_aqi': weekly_risk.min()
                    }
                
                # Monthly patterns
                if 'month' in data.columns:
                    monthly_risk = data.groupby('month')['aqi'].mean()
                    temporal_risk['monthly'] = {
                        'peak_month': monthly_risk.idxmax(),
                        'peak_aqi': monthly_risk.max(),
                        'low_month': monthly_risk.idxmin(),
                        'low_aqi': monthly_risk.min()
                    }
        
        except Exception as e:
            print(f"Error analyzing temporal risk patterns: {str(e)}")
        
        return temporal_risk
    
    def _classify_risk_level(self, risk_score, exceedance_factors):
        max_exceedance = max(exceedance_factors.values()) if exceedance_factors else 0
        
        # Combined classification based on risk score and exceedance
        if risk_score >= 7.0 or max_exceedance >= 3.0:
            return {
                'level': 'Critical',
                'color': '#8B0000',
                'description': 'Immediate health risks for all population groups'
            }
        elif risk_score >= 5.0 or max_exceedance >= 2.0:
            return {
                'level': 'High',
                'color': '#FF0000',
                'description': 'Significant health risks, especially for sensitive groups'
            }
        elif risk_score >= 3.0 or max_exceedance >= 1.5:
            return {
                'level': 'Moderate',
                'color': '#FF7E00',
                'description': 'Moderate health risks for sensitive individuals'
            }
        elif risk_score >= 1.0 or max_exceedance >= 1.0:
            return {
                'level': 'Low-Moderate',
                'color': '#FFFF00',
                'description': 'Minor health risks for very sensitive individuals'
            }
        else:
            return {
                'level': 'Low',
                'color': '#00E400',
                'description': 'Minimal health risks for general population'
            }
    
    def _count_days_exceeding_who(self, data):
        try:
            daily_data = data.resample('D').mean()
            days_exceeding = 0
            
            for pollutant, who_limit in self.who_limits.items():
                if pollutant in daily_data.columns:
                    pollutant_data = daily_data[pollutant].dropna()
                    exceeding_days = (pollutant_data > who_limit).sum()
                    days_exceeding = max(days_exceeding, exceeding_days)
            
            return int(days_exceeding)
            
        except Exception as e:
            print(f"Error counting days exceeding WHO limits: {str(e)}")
            return 0
    
    def _create_empty_risk_assessment(self, city_name):
        return {
            'city': city_name,
            'exceedance_factor': 0.0,
            'risk_score': 0.0,
            'exceedance_factors_by_pollutant': {},
            'daily_exceedances': {},
            'health_impact': {
                'overall_level': 'Unknown',
                'primary_concerns': [],
                'vulnerable_groups_risk': 'Unknown',
                'recommended_actions': []
            },
            'temporal_risk': {},
            'risk_classification': {
                'level': 'Unknown',
                'color': '#CCCCCC',
                'description': 'Unable to assess risk due to insufficient data'
            },
            'days_exceeding_who': 0,
            'population_at_risk': 0,
            'assessment_date': datetime.now().isoformat()
        }
    
    def compare_city_risks(self, risk_assessments):
        try:
            if not risk_assessments:
                return {}
            
            cities_data = []
            for city, assessment in risk_assessments.items():
                cities_data.append({
                    'city': city,
                    'risk_score': assessment.get('risk_score', 0),
                    'exceedance_factor': assessment.get('exceedance_factor', 0),
                    'days_exceeding_who': assessment.get('days_exceeding_who', 0),
                    'population_at_risk': assessment.get('population_at_risk', 0)
                })
            
            cities_df = pd.DataFrame(cities_data)
            
            # Rankings
            rankings = {
                'highest_risk_score': cities_df.nlargest(5, 'risk_score')[['city', 'risk_score']].to_dict('records'),
                'highest_exceedance': cities_df.nlargest(5, 'exceedance_factor')[['city', 'exceedance_factor']].to_dict('records'),
                'most_days_exceeding': cities_df.nlargest(5, 'days_exceeding_who')[['city', 'days_exceeding_who']].to_dict('records'),
                'largest_population_at_risk': cities_df.nlargest(5, 'population_at_risk')[['city', 'population_at_risk']].to_dict('records')
            }
            
            # Summary statistics
            summary_stats = {
                'total_cities_assessed': len(cities_df),
                'avg_risk_score': cities_df['risk_score'].mean(),
                'cities_exceeding_2x_who': len(cities_df[cities_df['exceedance_factor'] > 2.0]),
                'total_population_at_risk': cities_df['population_at_risk'].sum(),
                'most_concerning_city': cities_df.loc[cities_df['risk_score'].idxmax(), 'city'] if len(cities_df) > 0 else 'None'
            }
            
            return {
                'rankings': rankings,
                'summary_statistics': summary_stats,
                'detailed_comparison': cities_data
            }
            
        except Exception as e:
            print(f"Error comparing city risks: {str(e)}")
            return {}