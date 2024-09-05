import numpy as np
import pandas as pd
from scipy.stats import norm

def generate_gaussian_seasonality(date_range, peak_times, peak_amplitudes, peak_widths, frequency):
    """
    Generate a seasonal pattern using Gaussian functions.
    
    Parameters:
    - date_range: pd.DatetimeIndex, the time range for the series
    - peak_times: list of float, the times of the day where peaks occur
    - peak_amplitudes: list of float, the amplitudes of the peaks
    - peak_widths: list of float, the standard deviations of the peaks
    - frequency: str, frequency of the time series (e.g., 'H', '15min', 'T')
    
    Returns:
    - np.array, the generated seasonal pattern
    """
    seasonality = np.zeros(len(date_range))
    timedelta = pd.Timedelta(frequency).total_seconds() / 3600.0  # Convert to hours
    times = date_range.hour + date_range.minute / 60.0 + date_range.second / 3600.0 + date_range.microsecond / 3_600_000_000.0
    times /= timedelta
    for peak_time, peak_amplitude, peak_width in zip(peak_times, peak_amplitudes, peak_widths):
        seasonality += peak_amplitude * norm.pdf(times, peak_time, peak_width)
    return seasonality

def generate_demand(start_date, end_date, frequency='H', base_demand=50, 
                    daily_peak_times=[6, 18], daily_peak_amplitudes=[10, 15], 
                    daily_peak_widths=[2, 2], weekly_peak_days=[1, 5], 
                    weekly_peak_amplitudes=[5, 10], weekly_peak_widths=[1, 1], 
                    random_noise_std=5):
    """
    Generate an electrical demand time series.
    
    Parameters:
    - start_date: str, the start date of the time series (e.g., '2023-01-01')
    - end_date: str, the end date of the time series (e.g., '2023-12-31')
    - frequency: str, frequency of the time series (e.g., 'H', '15min', '1min')
    - base_demand: float, the base level of demand
    - daily_peak_times: list of float, times of the day where daily peaks occur
    - daily_peak_amplitudes: list of float, amplitudes of daily peaks
    - daily_peak_widths: list of float, standard deviations of daily peaks
    - weekly_peak_days: list of int, days of the week where weekly peaks occur (0=Monday, 6=Sunday)
    - weekly_peak_amplitudes: list of float, amplitudes of weekly peaks
    - weekly_peak_widths: list of float, standard deviations of weekly peaks
    - random_noise_std: float, standard deviation of random noise
    
    Returns:
    - pd.Series, the generated time series of electrical demand
    """
    # Generate time range
    date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
    
    # Create daily seasonality using Gaussian peaks
    daily_seasonality = generate_gaussian_seasonality(date_range, daily_peak_times, daily_peak_amplitudes, daily_peak_widths, frequency)
    
    # Create weekly seasonality using Gaussian peaks
    weekly_seasonality = np.zeros(len(date_range))
    days = date_range.dayofweek + (date_range.hour / 24.0) + (date_range.minute / (24.0 * 60.0)) + (date_range.second / (24.0 * 3600.0)) + (date_range.microsecond / (24.0 * 3_600_000_000.0))
    for peak_day, peak_amplitude, peak_width in zip(weekly_peak_days, weekly_peak_amplitudes, weekly_peak_widths):
        weekly_seasonality += peak_amplitude * norm.pdf(days, peak_day, peak_width)
    
    # Generate random noise
    random_noise = np.random.normal(0, random_noise_std, len(date_range))
    
    # Combine all components
    demand = base_demand + daily_seasonality + weekly_seasonality + random_noise
    
    # Create a pandas Series
    demand_series = pd.Series(demand, index=date_range)
    
    return demand_series



def generate_demand_matrix(start_date, end_date, column_dimension, frequency='H', base_demand=50, 
                           daily_peak_times=[6, 18], daily_peak_amplitudes=[10, 15], 
                           daily_peak_widths=[2, 2], weekly_peak_days=[1, 5], 
                           weekly_peak_amplitudes=[5, 10], weekly_peak_widths=[1, 1], 
                           random_noise_std=5):
    """
    Generate a 2-dimensional array of demands by calling generate_demand multiple times.
    
    Parameters:
    - start_date: str, the start date of the time series (e.g., '2023-01-01')
    - end_date: str, the end date of the time series (e.g., '2023-12-31')
    - column_dimension: int, the number of demand series to generate (number of columns)
    - frequency: str, frequency of the time series (e.g., 'H', '15min', '1min')
    - base_demand: float, the base level of demand
    - daily_peak_times: list of float, times of the day where daily peaks occur
    - daily_peak_amplitudes: list of float, amplitudes of daily peaks
    - daily_peak_widths: list of float, standard deviations of daily peaks
    - weekly_peak_days: list of int, days of the week where weekly peaks occur (0=Monday, 6=Sunday)
    - weekly_peak_amplitudes: list of float, amplitudes of weekly peaks
    - weekly_peak_widths: list of float, standard deviations of weekly peaks
    - random_noise_std: float, standard deviation of random noise
    
    Returns:
    - np.array, a 2-dimensional array where each column is a generated demand series
    """
    demand_matrix = []
    
    for _ in range(column_dimension):
        demand_series = generate_demand(start_date, end_date, frequency, base_demand, 
                                        daily_peak_times, daily_peak_amplitudes, daily_peak_widths, 
                                        weekly_peak_days, weekly_peak_amplitudes, weekly_peak_widths, 
                                        random_noise_std)
        demand_matrix.append(demand_series.values)
    
    return np.array(demand_matrix).T


def generate_demand_df(start_date, end_date, frequency, indices):
    """
    Generate a DataFrame of demand time series.
    
    Parameters:
    - start_date: str, the start date of the time series (e.g., '2023-01-01')
    - end_date: str, the end date of the time series (e.g., '2023-12-31')
    - frequency: str, frequency of the time series (e.g., 'H', '15min', '1min')
    - indices: list of str, the indices for the columns of the DataFrame
    
    Returns:
    - pd.DataFrame, the generated DataFrame of demand time series
    """
    demand_matrix = generate_demand_matrix(start_date, end_date, len(indices), frequency)
    return pd.DataFrame(demand_matrix, columns=indices, index=pd.date_range(start_date, end_date, freq=frequency))


def sunrise_sunset_time(day_of_year):
    """
    Approximate the sunrise and sunset time based on the day of the year.
    """
    # Approximate sunrise and sunset times in hours (0-24)
    # Assume sunrise is earliest at day 172 (June 21, summer solstice) and latest at day 355 (Dec 21, winter solstice)
    sunrise = 6 + 2 * np.cos(2 * np.pi * (day_of_year - 172) / 365)
    sunset = 18 - 2 * np.cos(2 * np.pi * (day_of_year - 172) / 365)
    return sunrise, sunset

def generate_daily_irradiance_pattern(date_range, frequency='1H'):
    """
    Generate a daily solar irradiance pattern.
    """
    irradiance = np.zeros(len(date_range))
    for i, current_time in enumerate(date_range):
        day_of_year = current_time.timetuple().tm_yday
        sunrise, sunset = sunrise_sunset_time(day_of_year)
        
        # Convert times to fractions of the day
        current_hour = current_time.hour + current_time.minute / 60.0
        
        # Define a Gaussian-like peak around noon
        if sunrise <= current_hour <= sunset:
            peak_hour = (sunrise + sunset) / 2
            width = (sunset - sunrise) / 4
            irradiance[i] = np.exp(-((current_hour - peak_hour) ** 2) / (2 * width ** 2))
    
    return irradiance

def generate_cloud_cover_pattern(date_range, base_coverage=0.5, variation=0.3, frequency='1H'):
    """
    Generate a cloud cover pattern.
    
    Parameters:
    - date_range: pd.DatetimeIndex, the time range for the series
    - base_coverage: float, the base level of cloud cover (0 to 1)
    - variation: float, the amplitude of cloud cover variation (0 to 1)
    
    Returns:
    - np.array, the generated cloud cover pattern
    """
    np.random.seed(0)  # For reproducibility
    daily_variation = variation * np.sin(2 * np.pi * (date_range.hour / 24.0))
    random_variation = variation * np.random.normal(0, 0.1, len(date_range))
    cloud_cover = base_coverage + daily_variation + random_variation
    cloud_cover = np.clip(cloud_cover, 0, 1)  # Ensure values are between 0 and 1
    return cloud_cover

def generate_pv_power(start_date, end_date, frequency='1H', base_power=100, 
                      base_coverage=0.5, variation=0.3):
    """
    Generate the maximum available power of a PV generator considering year seasonality and cloud obstruction.
    
    Parameters:
    - start_date: str, the start date of the time series (e.g., '2023-01-01')
    - end_date: str, the end date of the time series (e.g., '2023-12-31')
    - frequency: str, frequency of the time series (e.g., '1H', '15T', '1T')
    - base_power: float, the base level of PV power generation
    - base_coverage: float, the base level of cloud cover (0 to 1)
    - variation: float, the amplitude of cloud cover variation (0 to 1)
    
    Returns:
    - pd.Series, the generated time series of PV power generation
    """
    # Generate time range
    date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
    
    # Generate daily irradiance pattern
    daily_irradiance = generate_daily_irradiance_pattern(date_range, frequency)
    
    # Generate cloud cover pattern
    cloud_cover = generate_cloud_cover_pattern(date_range, base_coverage, variation)
    
    # Calculate the available power considering cloud cover
    available_power = base_power * daily_irradiance * (1 - cloud_cover)
    
    # Ensure power is non-negative
    available_power = np.maximum(available_power, 0)
    
    # Create a pandas Series
    power_series = pd.Series(available_power, index=date_range)
    
    return power_series


def generate_pv_power_matrix(start_date, end_date, column_dimension, frequency='1H', base_power=100, 
                             base_coverage=0.5, variation=0.3):
    """
    Generate a 2-dimensional array of PV power series by calling generate_pv_power multiple times.
    
    Parameters:
    - start_date: str, the start date of the time series (e.g., '2023-01-01')
    - end_date: str, the end date of the time series (e.g., '2023-12-31')
    - column_dimension: int, the number of PV power series to generate (number of columns)
    - frequency: str, frequency of the time series (e.g., '1H', '15T', '1T')
    - base_power: float, the base level of PV power generation
    - base_coverage: float, the base level of cloud cover (0 to 1)
    - variation: float, the amplitude of cloud cover variation (0 to 1)
    
    Returns:
    - np.array, a 2-dimensional array where each column is a generated PV power series
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
    power_matrix = np.zeros((len(date_range), column_dimension))
    
    for i in range(column_dimension):
        power_series = generate_pv_power(start_date, end_date, frequency, base_power, 
                                         base_coverage, variation)
        power_matrix[:, i] = power_series.values
    
    return power_matrix

def generate_pv_power_df(start_date, end_date, frequency, indices):
    """
    Generate a DataFrame of PV power time series.
    
    Parameters:
    - start_date: str, the start date of the time series (e.g., '2023-01-01')
    - end_date: str, the end date of the time series (e.g., '2023-12-31')
    - frequency: str, frequency of the time series (e.g., '1H', '15T', '1T')
    - indices: list of str, the indices for the columns of the DataFrame
    
    Returns:
    - pd.DataFrame, the generated DataFrame of PV power time series
    """
    power_matrix = generate_pv_power_matrix(start_date, end_date, len(indices), frequency)
    return pd.DataFrame(power_matrix, columns=indices, index=pd.date_range(start_date, end_date, freq=frequency))



def generate_solar_irradiance(start_date, end_date, column_dimension, frequency='1H', base_power=100, 
                             base_coverage=0.5, variation=0.3):
    """
    Generate a 2-dimensional array of PV power series by calling generate_pv_power multiple times.
    
    Parameters:
    - start_date: str, the start date of the time series (e.g., '2023-01-01')
    - end_date: str, the end date of the time series (e.g., '2023-12-31')
    - column_dimension: int, the number of PV power series to generate (number of columns)
    - frequency: str, frequency of the time series (e.g., '1H', '15T', '1T')
    - base_power: float, the base level of PV power generation
    - base_coverage: float, the base level of cloud cover (0 to 1)
    - variation: float, the amplitude of cloud cover variation (0 to 1)
    
    Returns:
    - np.array, a 2-dimensional array where each column is a generated PV power series
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
    power_matrix = np.zeros((len(date_range), column_dimension))
    
    for i in range(column_dimension):
        power_series = generate_pv_power(start_date, end_date, frequency, base_power, 
                                         base_coverage, variation)
        power_matrix[:, i] = power_series.values
    
    return power_matrix