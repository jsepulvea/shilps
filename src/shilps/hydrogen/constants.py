
class Constants:
    # Universal constants
    R = 8.314  # Universal gas constant in J/(mol·K)
    F = 96485  # Faraday constant in C/mol

    # Antoine coefficients for water (A, B, C)
    ANTOINE_COEFFICIENTS_WATER = {
        'A': 8.07131,
        'B': 1730.63,
        'C': 233.426
    }



def celsius_to_kelvin(celsius):
    return celsius + 273.15