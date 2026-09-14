"""Weather, via Open-Meteo. No API key, no account."""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, Field

from jarvis.errors import ToolError
from jarvis.logging_setup import get_logger
from jarvis.tools.registry import ToolContext, tool

log = get_logger("tools.weather")

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 10.0

# Open-Meteo reports conditions as WMO codes. Phrased for speech.
WMO = {
    0: "clear",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "freezing fog",
    51: "drizzling lightly",
    53: "drizzling",
    55: "drizzling heavily",
    56: "freezing drizzle",
    57: "heavy freezing drizzle",
    61: "raining lightly",
    63: "raining",
    65: "raining heavily",
    66: "freezing rain",
    67: "heavy freezing rain",
    71: "snowing lightly",
    73: "snowing",
    75: "snowing heavily",
    77: "snow grains",
    80: "with light showers",
    81: "with showers",
    82: "with heavy showers",
    85: "with snow showers",
    86: "with heavy snow showers",
    95: "thundery",
    96: "thundery with hail",
    99: "thundery with heavy hail",
}


class WeatherArgs(BaseModel):
    location: str | None = Field(
        None, description="Place name, e.g. 'Edinburgh'. Omit for the configured default."
    )


@tool(
    name="get_weather",
    description=(
        "Current weather and today's forecast for a place. Use whenever the user asks "
        "about weather, temperature, or whether they need a coat."
    ),
    args=WeatherArgs,
)
def get_weather(args: WeatherArgs, ctx: ToolContext) -> str:
    cfg = ctx.config.tools.weather
    metric = ctx.config.general.units == "metric"

    if args.location:
        latitude, longitude, name = _geocode(args.location)
    elif cfg.default_latitude is not None and cfg.default_longitude is not None:
        latitude, longitude = cfg.default_latitude, cfg.default_longitude
        name = cfg.default_location_name or "your location"
    else:
        raise ToolError(
            "No location was given and no default is configured. Ask the user which "
            "place they mean."
        )

    data = _fetch(latitude, longitude, metric)
    return _describe(data, name, metric)


def _geocode(place: str) -> tuple[float, float, str]:
    payload = _get(GEOCODE_URL, {"name": place, "count": 1, "language": "en", "format": "json"})
    results = payload.get("results") or []
    if not results:
        raise ToolError(f"I couldn't find anywhere called {place!r}.")
    top = results[0]
    label = top.get("name", place)
    country = top.get("country")
    return (
        float(top["latitude"]),
        float(top["longitude"]),
        (f"{label}, {country}" if country and country != label else label),
    )


def _fetch(latitude: float, longitude: float, metric: bool) -> dict[str, Any]:
    return _get(
        FORECAST_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "forecast_days": 1,
            "timezone": "auto",
            "temperature_unit": "celsius" if metric else "fahrenheit",
            "wind_speed_unit": "kmh" if metric else "mph",
        },
    )


def _get(url: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        response = httpx.get(url, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        return dict(response.json())
    except httpx.TimeoutException as exc:
        raise ToolError("The weather service didn't answer in time.") from exc
    except httpx.HTTPError as exc:
        raise ToolError(f"The weather service is unreachable ({exc}).") from exc
    except ValueError as exc:
        raise ToolError("The weather service returned something unreadable.") from exc


def _describe(data: dict[str, Any], name: str, metric: bool) -> str:
    current = data.get("current") or {}
    daily = data.get("daily") or {}
    degrees = "C" if metric else "F"

    temperature = current.get("temperature_2m")
    feels = current.get("apparent_temperature")
    condition = WMO.get(int(current.get("weather_code", -1)), "unsettled")
    wind = current.get("wind_speed_10m")

    parts = [f"In {name} it is {_round(temperature)} degrees {degrees} and {condition}"]
    if feels is not None and temperature is not None and abs(feels - temperature) >= 2:
        parts.append(f"feels like {_round(feels)}")
    if wind is not None and wind >= (20 if metric else 12):
        parts.append(f"windy at {_round(wind)} {'kilometres' if metric else 'miles'} an hour")

    highs = (daily.get("temperature_2m_max") or [None])[0]
    lows = (daily.get("temperature_2m_min") or [None])[0]
    if highs is not None and lows is not None:
        parts.append(f"today between {_round(lows)} and {_round(highs)}")

    rain = (daily.get("precipitation_probability_max") or [None])[0]
    if rain is not None and rain >= 30:
        parts.append(f"{int(rain)} percent chance of rain")

    return ", ".join(parts) + "."


def _round(value: float | None) -> str:
    return "unknown" if value is None else str(round(value))
