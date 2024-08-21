# thermodynamics.py
import math
from .constants import *



def antoine_pressure(temp_celsius:float, A:float, B:float, C:float):
    """
    Calculate the saturation pressure using the Antoine equation.
    
    Parameters:
    - temp_celsius: Temperature in degrees Celsius
    - A: Antoine coefficient A
    - B: Antoine coefficient B
    - C: Antoine coefficient C
    
    Returns:
    - Saturation pressure in bar
    """
    # Antoine equation in mmHg
    log10_P_sat = A - (B / (C + temp_celsius))
    P_sat_mmHg = 10 ** log10_P_sat
    
    # Convert mmHg to bar (1 bar = 750.06 mmHg)
    P_sat_bar = P_sat_mmHg / 750.06
    return P_sat_bar


def partial_pressure_estimator_H2(P_total, temp_celsius):
    """
    Estimate the partial pressure of H2 from the total pressure and temperature.
    
    Parameters:
    - P_total: Total pressure in bar
    - temp_celsius: Temperature in degrees Celsius
    
    Returns:
    - Partial pressure of H2 in bar
    """
    # Antoine coefficients for water
    A = 8.07131
    B = 1730.63
    C = 233.426
    
    # Calculate saturation pressure of water vapor
    P_H2O = antoine_pressure(temp_celsius, A, B, C)
    
    # Effective total pressure for gases (subtract water vapor pressure)
    P_effective = P_total - P_H2O
    
    # Calculate partial pressure of H2 based on stoichiometry
    P_H2 = (2/3) * P_effective
    return P_H2


def partial_pressure_estimator_O2(P_total, temp_celsius):
    """
    Estimate the partial pressure of O2 from the total pressure and temperature.
    
    Parameters:
    - P_total: Total pressure in bar
    - temp_celsius: Temperature in degrees Celsius
    
    Returns:
    - Partial pressure of O2 in bar
    """
    # Antoine coefficients for water
    A = 8.07131
    B = 1730.63
    C = 233.426
    
    # Calculate saturation pressure of water vapor
    P_H2O = antoine_pressure(temp_celsius, A, B, C)
    
    # Effective total pressure for gases (subtract water vapor pressure)
    P_effective = P_total - P_H2O
    
    # Calculate partial pressure of O2 based on stoichiometry
    P_O2 = (1/3) * P_effective
    return P_O2