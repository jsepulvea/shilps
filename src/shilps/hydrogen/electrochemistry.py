import math
from .constants import *


def standard_reversible_voltage(T_kelvin):
    """
    Calculate the standard reversible voltage as a function of temperature.
    
    Parameters:
    - T_kelvin: Temperature in Kelvin
    
    Returns:
    - Standard reversible voltage in volts
    """
    V0 = (1.5184 
          - 1.5421e-3 * T_kelvin 
          + 9.523e-5 * T_kelvin * math.log(T_kelvin) 
          + 9.84e-8 * T_kelvin**2)
    return V0


def nernst_potential(T_celsius, P_H2, P_O2, P_H2O):
    """
    Calculate the Nernst potential.
    
    Parameters:
    - T_celsius: Temperature in degrees Celsius
    - P_H2: Partial pressure of hydrogen in bar
    - P_O2: Partial pressure of oxygen in bar
    - P_H2O: Partial pressure of water in bar
    
    Returns:
    - Nernst potential in volts
    """
    T_kelvin = celsius_to_kelvin(T_celsius)
    V0 = standard_reversible_voltage(T_kelvin)
    n = 2  # For the electrolysis of water, 2 moles of electrons are transferred
    
    RT_over_nF = (Constants.R * T_kelvin) / (n * Constants.F)
    ln_argument = (P_H2 * P_O2 ** 0.5) / P_H2O
    
    V_ref = V0 + (RT_over_nF * math.log(ln_argument))
    return V_ref


def membrane_conductivity(T_kelvin, lambda_mem):
    """
    Calculate the membrane conductivity.
    
    Parameters:
    - T_kelvin: Temperature in Kelvin
    - lambda_mem: Humidification of the membrane
    
    Returns:
    - Membrane conductivity in S/cm
    """
    sigma_mem = (0.005139 * lambda_mem - 0.00326) * math.exp(1268 * (1 / 303 - 1 / T_kelvin))
    return sigma_mem


def membrane_resistance(delta_mem, T_celsius, lambda_mem):
    """
    Calculate the membrane resistance.
    
    Parameters:
    - delta_mem: Thickness of the membrane in cm
    - T_celsius: Temperature in degrees Celsius
    - lambda_mem: Humidification of the membrane
    
    Returns:
    - Membrane resistance in ohms
    """
    T_kelvin = T_celsius + 273.15
    sigma_mem = membrane_conductivity(T_kelvin, lambda_mem)
    R_mem = delta_mem / sigma_mem
    return R_mem


