from __future__ import annotations


def c_to_f(celsius: float) -> float:
    """Converts Celsius to Fahrenheit, rounded to 1 decimal place."""
    return round(celsius * 9.0 / 5.0 + 32.0, 1)


def format_temp_c(celsius: float | None) -> str:
    """Renders a Celsius value with its Fahrenheit equivalent, e.g. "24.0°C (75.2°F)".

    All temperatures sourced from Open-Meteo throughout this app are in Celsius (the API's
    default unit) -- this is the single place that adds the Fahrenheit conversion so every
    display shows both units consistently.
    """
    if celsius is None:
        return "unknown"
    return f"{celsius:g}°C ({c_to_f(celsius):g}°F)"
