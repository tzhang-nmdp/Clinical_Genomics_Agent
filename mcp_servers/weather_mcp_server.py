from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP
import certifi

# Initialize FastMCP server
mcp = FastMCP("weather")

# Constants
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"

async def make_nws_request(url: str) -> tuple[dict[str, Any] | None, str | None]:
    """Make a request to the NWS API with proper error handling."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
    async with httpx.AsyncClient(verify=False) as client:
    # async with httpx.AsyncClient(verify=certifi.where()) as client:        
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json(), None
        except httpx.HTTPStatusError as e:
            return None, f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        except httpx.TimeoutException:
            return None, "Request timed out."
        except Exception as e:
            return None, str(e)


def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    props = feature["properties"]
    return f"""
Event: {props.get("event", "Unknown")}
Area: {props.get("areaDesc", "Unknown")}
Severity: {props.get("severity", "Unknown")}
Description: {props.get("description", "No description available")}
Instructions: {props.get("instruction", "No specific instructions provided")}
"""

@mcp.tool()
async def get_alerts(state: str) -> str:
    """Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data, err = await make_nws_request(url)

    if not data or "features" not in data:
        return f"Unable to fetch alerts or no alerts found. {err or ''}".strip()

    if not data["features"]:
        return "No active alerts for this state."

    alerts = [format_alert(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)


@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """Get weather forecast for a location.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    """
    # NWS API only covers the United States
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return "Invalid coordinates."
    if not (24 <= latitude <= 50 and -125 <= longitude <= -66):
        return "The NWS weather API only supports locations within the United States."

    # First get the forecast grid endpoint
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data, err = await make_nws_request(points_url)

    if not points_data:
        return f"Unable to fetch forecast data for this location. {err or ''}".strip()

    # Get the forecast URL from the points response
    forecast_url = points_data["properties"]["forecast"]
    forecast_data, err = await make_nws_request(forecast_url)

    if not forecast_data:
        return f"Unable to fetch detailed forecast. {err or ''}".strip()

    # Format the periods into a readable forecast
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]:  # Only show next 5 periods
        forecast = f"""
{period["name"]}:
Temperature: {period["temperature"]}°{period["temperatureUnit"]}
Wind: {period["windSpeed"]} {period["windDirection"]}
Forecast: {period["detailedForecast"]}
"""
        forecasts.append(forecast)

    return "\n---\n".join(forecasts)

def main():
    # Initialize and run the server
    # mcp.run(transport="stdio")
    mcp.run()

if __name__ == "__main__":
    main()
