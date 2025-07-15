from tool import Tool
import requests
import json
from datetime import datetime
from typing import Dict, Optional, Union
import os

class Weather(Tool):
    """
    A class representing a weather tool in the Hozie system.
    """
    def __init__(self):
        """
        Initialize the WeatherFetcher with your OpenWeatherMap API key
        """
        self.api_key = os.environ.get("WEATHER_API_KEY","key")
        self.base_url = "https://api.openweathermap.org/data/2.5"
    
    def get_current_weather(self, city: str, units: str = "metric") -> Dict:
        """
        Fetch current weather data for a city
        
        Args:
            city: Name of the city (e.g., "London" or "London,UK")
            units: Temperature units - "metric" (Celsius), "imperial" (Fahrenheit), or "kelvin"
        
        Returns:
            Dictionary containing weather data
        """
        endpoint = f"{self.base_url}/weather"
        
        params = {
            "q": city,
            "appid": self.api_key,
            "units": units
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract and format the relevant weather information
            weather_data = {
                "location": {
                    "city": data["name"],
                    "country": data["sys"]["country"],
                    "coordinates": {
                        "lat": data["coord"]["lat"],
                        "lon": data["coord"]["lon"]
                    }
                },
                "current": {
                    "temperature": data["main"]["temp"],
                    "feels_like": data["main"]["feels_like"],
                    "temp_min": data["main"]["temp_min"],
                    "temp_max": data["main"]["temp_max"],
                    "pressure": data["main"]["pressure"],
                    "humidity": data["main"]["humidity"]
                },
                "weather": {
                    "main": data["weather"][0]["main"],
                    "description": data["weather"][0]["description"],
                    "icon": data["weather"][0]["icon"]
                },
                "wind": {
                    "speed": data["wind"]["speed"],
                    "direction": data["wind"].get("deg", None),
                    "gust": data["wind"].get("gust", None)
                },
                "visibility": data.get("visibility", None),
                "clouds": data["clouds"]["all"],
                "timestamp": datetime.fromtimestamp(data["dt"]).isoformat(),
                "timezone": data["timezone"],
                "units": units
            }
            
            return weather_data
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                raise ValueError(f"City '{city}' not found")
            elif response.status_code == 401:
                raise ValueError("Invalid API key")
            else:
                raise Exception(f"HTTP error occurred: {e}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching weather data: {e}")
    
    def get_weather_by_coordinates(self, lat: float, lon: float, units: str = "metric") -> Dict:
        """
        Fetch weather data by geographic coordinates
        
        Args:
            lat: Latitude
            lon: Longitude
            units: Temperature units
        
        Returns:
            Dictionary containing weather data
        """
        endpoint = f"{self.base_url}/weather"
        
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": units
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return self._process_weather_response(response.json(), units)
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching weather data: {e}")
    
    def get_forecast(self, city: str, units: str = "metric", cnt: int = 5) -> Dict:
        """
        Fetch 5-day weather forecast (3-hour intervals)
        
        Args:
            city: Name of the city
            units: Temperature units
            cnt: Number of timestamps to return (max 40)
        
        Returns:
            Dictionary containing forecast data
        """
        endpoint = f"{self.base_url}/forecast"
        
        params = {
            "q": city,
            "appid": self.api_key,
            "units": units,
            "cnt": min(cnt, 40)  # API limit is 40
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            forecast_data = {
                "city": {
                    "name": data["city"]["name"],
                    "country": data["city"]["country"],
                    "coordinates": {
                        "lat": data["city"]["coord"]["lat"],
                        "lon": data["city"]["coord"]["lon"]
                    }
                },
                "forecasts": []
            }
            
            for item in data["list"]:
                forecast_data["forecasts"].append({
                    "datetime": datetime.fromtimestamp(item["dt"]).isoformat(),
                    "temperature": item["main"]["temp"],
                    "feels_like": item["main"]["feels_like"],
                    "temp_min": item["main"]["temp_min"],
                    "temp_max": item["main"]["temp_max"],
                    "pressure": item["main"]["pressure"],
                    "humidity": item["main"]["humidity"],
                    "weather": item["weather"][0]["description"],
                    "wind_speed": item["wind"]["speed"],
                    "clouds": item["clouds"]["all"],
                    "precipitation": item.get("rain", {}).get("3h", 0)
                })
            
            return forecast_data
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching forecast data: {e}")
    
    def _process_weather_response(self, data: Dict, units: str) -> Dict:
        """Helper method to process weather API response"""
        return {
            "location": {
                "city": data["name"],
                "country": data["sys"]["country"],
                "coordinates": {
                    "lat": data["coord"]["lat"],
                    "lon": data["coord"]["lon"]
                }
            },
            "current": {
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "pressure": data["main"]["pressure"],
                "humidity": data["main"]["humidity"]
            },
            "weather": {
                "description": data["weather"][0]["description"]
            },
            "wind": {
                "speed": data["wind"]["speed"]
            },
            "timestamp": datetime.fromtimestamp(data["dt"]).isoformat(),
            "units": units
        }


# Standalone function for simple usage
def get_weather(city: str, api_key: str, units: str = "metric") -> Dict:
    """
    Simple function to get current weather for a city
    
    Args:
        city: City name
        api_key: OpenWeatherMap API key
        units: Temperature units ("metric", "imperial", or "kelvin")
    
    Returns:
        Dictionary with weather data
    """
    url = f"https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": units
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        return {
            "city": data["name"],
            "country": data["sys"]["country"],
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "description": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"],
            "timestamp": datetime.now().isoformat()
        }
        
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            return {"error": f"City '{city}' not found"}
        elif response.status_code == 401:
            return {"error": "Invalid API key"}
        else:
            return {"error": f"HTTP error: {str(e)}"}
    except Exception as e:
        return {"error": f"Error fetching weather: {str(e)}"}


# Utility functions
def format_weather_report(weather_data: Dict) -> str:
    """
    Format weather data into a readable report
    
    Args:
        weather_data: Weather data dictionary
    
    Returns:
        Formatted weather report string
    """
    if "error" in weather_data:
        return f"Error: {weather_data['error']}"
    
    if "current" in weather_data:  # Using class method output
        data = weather_data
        location = f"{data['location']['city']}, {data['location']['country']}"
        temp = data['current']['temperature']
        feels = data['current']['feels_like']
        humidity = data['current']['humidity']
        desc = data['weather']['description']
        wind = data['wind']['speed']
        units_symbol = "°C" if data.get('units') == 'metric' else "°F"
    else:  # Using simple function output
        data = weather_data
        location = f"{data['city']}, {data['country']}"
        temp = data['temperature']
        feels = data['feels_like']
        humidity = data['humidity']
        desc = data['description']
        wind = data['wind_speed']
        units_symbol = "°C"  # Default to metric
    
    report = f"""
Weather Report for {location}
{'=' * 40}
Temperature: {temp}{units_symbol}
Feels Like: {feels}{units_symbol}
Conditions: {desc.title()}
Humidity: {humidity}%
Wind Speed: {wind} m/s
Last Updated: {data['timestamp']}
    """
    
    return report.strip()


# Example usage
if __name__ == "__main__":
    # Replace with your actual API key

    
    # Example 2: Using the class for more features
    print("Example 2: Using Weather class")
    weather = Weather()
    
    # Get current weather
    try:
        current = weather.get_current_weather("New York")
        print(format_weather_report(current))
        
        # Get weather by coordinates (Tokyo)
        tokyo_weather = weather.get_weather_by_coordinates(35.6762, 139.6503)
        print(f"\nWeather in Tokyo: {tokyo_weather['current']['temperature']}°C")
        
        # Get forecast
        forecast = weather.get_forecast("Paris", cnt=3)
        print(f"\nForecast for {forecast['city']['name']}:")
        for f in forecast['forecasts']:
            print(f"  {f['datetime']}: {f['temperature']}°C - {f['weather']}")
            
    except Exception as e:
        print(f"Error: {e}")
    