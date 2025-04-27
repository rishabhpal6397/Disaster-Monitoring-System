import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime
import time
import pymongo
from pymongo import MongoClient
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def connect_to_mongodb():
    """Connect to MongoDB database"""
    try:
        client = MongoClient("mongodb://localhost:27017/")
        db_name = os.environ.get("DisasterDataConc","DisasterData")
        db = client[db_name]
        return db
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        return None

def fetch_emdat_data():
    """
    Fetch disaster data from EM-DAT database
    This is a simulation of what would be done with actual API access
    """
    logging.info("Starting EM-DAT data fetch process")
    
    # Connect to MongoDB
    db = connect_to_mongodb()
    if db is None:
        logging.error("Database connection failed")
        return None
    
    # Check if collection exists and has data
    if "disasters" in db.list_collection_names() and db.disasters.count_documents({}) > 0:
        logging.info("Data already exists in database, skipping fetch")
        return True
    
    try:
        # In a real implementation, this would use the EM-DAT API with authentication
        username = "rishavpal309@gmail.com"
        password = "@Rishabh6397"
        
        if not username or not password:
            logging.error("EM-DAT credentials not found")
            return False
        
        logging.info("Simulating EM-DAT data fetch (would use API in production)")
        
        # Generate sample data based on EM-DAT structure
        # In a real implementation, this would be replaced with actual API calls
        
        # Start and end years
        start_year = 2005
        end_year = 2025
        
        # Generate data
        data = generate_sample_disaster_data(start_year, end_year)
        
        # Store in MongoDB
        if len(data) > 0:
            def convert_np_strings(record):
                """Recursively convert np.str_ to regular Python strings in a dictionary"""
                for key, value in record.items():
                    if isinstance(value, np.str_):
                        record[key] = str(value)
                    elif isinstance(value, list):  # If value is a list, check each element
                        record[key] = [str(item) if isinstance(item, np.str_) else item for item in value]
                return record

            for i in range(len(data)):
                data[i] = convert_np_strings(data[i])
            
            db.disasters.insert_many(data)
            logging.info(f"Successfully stored {len(data)} disaster records in database")
            return True
        else:
            logging.error("No data generated")
            return False
            
    except Exception as e:
        logging.error(f"Error in fetch process: {e}")
        return False

def generate_sample_disaster_data(start_year, end_year):
    """Generate realistic sample disaster data"""
    logging.info(f"Generating sample data from {start_year} to {end_year}")
    
    # List of countries and regions
    countries = {
        "United States": {"region": "Americas", "iso3": "USA"},
        "China": {"region": "Asia", "iso3": "CHN"},
        "India": {"region": "Asia", "iso3": "IND"},
        "Japan": {"region": "Asia", "iso3": "JPN"},
        "Indonesia": {"region": "Asia", "iso3": "IDN"},
        "Philippines": {"region": "Asia", "iso3": "PHL"},
        "Brazil": {"region": "Americas", "iso3": "BRA"},
        "Mexico": {"region": "Americas", "iso3": "MEX"},
        "Italy": {"region": "Europe", "iso3": "ITA"},
        "France": {"region": "Europe", "iso3": "FRA"},
        "Germany": {"region": "Europe", "iso3": "DEU"},
        "United Kingdom": {"region": "Europe", "iso3": "GBR"},
        "Australia": {"region": "Oceania", "iso3": "AUS"},
        "New Zealand": {"region": "Oceania", "iso3": "NZL"},
        "South Africa": {"region": "Africa", "iso3": "ZAF"},
        "Nigeria": {"region": "Africa", "iso3": "NGA"},
        "Egypt": {"region": "Africa", "iso3": "EGY"},
        "Kenya": {"region": "Africa", "iso3": "KEN"},
        "Bangladesh": {"region": "Asia", "iso3": "BGD"},
        "Pakistan": {"region": "Asia", "iso3": "PAK"}
    }
    
    # Disaster types and subtypes with realistic coordinates
    disaster_types = {
        "Earthquake": {
            "group": "Geophysical",
            "countries": ["Japan", "Indonesia", "Italy", "Mexico", "China", "United States"],
            "severity_factor": 3.0
        },
        "Tsunami": {
            "group": "Geophysical",
            "countries": ["Japan", "Indonesia", "Philippines", "India"],
            "severity_factor": 4.0
        },
        "Volcanic Eruption": {
            "group": "Geophysical",
            "countries": ["Indonesia", "Philippines", "Italy", "Japan"],
            "severity_factor": 2.5
        },
        "Flood": {
            "group": "Hydrological",
            "countries": ["China", "India", "Bangladesh", "United States", "Brazil"],
            "severity_factor": 2.0
        },
        "Flash Flood": {
            "group": "Hydrological",
            "countries": ["United States", "India", "China", "France", "Germany"],
            "severity_factor": 1.8
        },
        "Hurricane/Cyclone": {
            "group": "Meteorological",
            "countries": ["United States", "Philippines", "Japan", "Mexico", "Bangladesh"],
            "severity_factor": 3.5
        },
        "Tornado": {
            "group": "Meteorological",
            "countries": ["United States", "Bangladesh", "Brazil"],
            "severity_factor": 2.0
        },
        "Wildfire": {
            "group": "Climatological",
            "countries": ["United States", "Australia", "Brazil", "Indonesia", "South Africa"],
            "severity_factor": 1.5
        },
        "Drought": {
            "group": "Climatological",
            "countries": ["Australia", "United States", "China", "India", "Brazil", "Kenya", "South Africa"],
            "severity_factor": 1.2
        },
        "Epidemic": {
            "group": "Biological",
            "countries": list(countries.keys()),  # All countries
            "severity_factor": 2.0
        },
        "Landslide": {
            "group": "Hydrological",
            "countries": ["China", "India", "Philippines", "Indonesia", "Italy"],
            "severity_factor": 1.7
        }
    }
    
    # Country coordinates (approximate centers)
    country_coords = {
        "United States": (39.8283, -98.5795),
        "China": (35.8617, 104.1954),
        "India": (20.5937, 78.9629),
        "Japan": (36.2048, 138.2529),
        "Indonesia": (-0.7893, 113.9213),
        "Philippines": (12.8797, 121.7740),
        "Brazil": (-14.2350, -51.9253),
        "Mexico": (23.6345, -102.5528),
        "Italy": (41.8719, 12.5674),
        "France": (46.2276, 2.2137),
        "Germany": (51.1657, 10.4515),
        "United Kingdom": (55.3781, -3.4360),
        "Australia": (-25.2744, 133.7751),
        "New Zealand": (-40.9006, 174.8860),
        "South Africa": (-30.5595, 22.9375),
        "Nigeria": (9.0820, 8.6753),
        "Egypt": (26.8206, 30.8025),
        "Kenya": (-0.0236, 37.9062),
        "Bangladesh": (23.6850, 90.3563),
        "Pakistan": (30.3753, 69.3451)
    }
    
    # Generate data
    data = []
    disaster_id = 1
    
    for year in range(start_year, end_year + 1):
        # Number of disasters varies by year with an increasing trend
        base_disasters = 300  # Base number of disasters per year
        yearly_increase = (year - start_year) * 10  # Increase over time
        random_variation = np.random.randint(-50, 50)  # Random variation
        
        num_disasters = base_disasters + yearly_increase + random_variation
        
        for _ in range(num_disasters):
            # Select random disaster type
            disaster_type = np.random.choice(list(disaster_types.keys()))
            disaster_info = disaster_types[disaster_type]
            
            # Select country based on disaster type's common countries
            country = np.random.choice(disaster_info["countries"])
            country_info = countries[country]
            
            # Get base coordinates for the country
            base_lat, base_lon = country_coords[country]
            
            # Add some random variation to coordinates
            lat = base_lat + np.random.uniform(-5, 5)
            lon = base_lon + np.random.uniform(-5, 5)
            
            # Generate dates
            start_month = np.random.randint(1, 13)
            start_day = np.random.randint(1, 29)
            duration = np.random.randint(1, 30) if disaster_type != "Drought" else np.random.randint(30, 180)
            
            start_date = f"{year}-{start_month:02d}-{start_day:02d}"
            
            # Now correctly create a datetime object
            start_datetime = datetime(year, start_month, start_day)
            end_datetime = start_datetime + pd.Timedelta(days=duration)
            
            # Format dates as strings
            start_date = start_datetime.strftime("%Y-%m-%d")
            end_date = end_datetime.strftime("%Y-%m-%d")

            
            # Generate severity and impact data
            severity_factor = disaster_info["severity_factor"]
            
            # More severe disasters are less frequent
            if np.random.random() < 0.8:
                severity_factor *= 0.5
            
            # Generate disaster record
            record = {
                "disaster_id": f"EM-DAT-{year}-{disaster_id:04d}",
                "year": year,
                "start_date": start_date,
                "end_date": end_date,
                "disaster_group": disaster_info["group"],
                "disaster_type": disaster_type,
                "country": country,
                "iso3": country_info["iso3"],
                "region": country_info["region"],
                "latitude": lat,
                "longitude": lon,
                "magnitude": np.random.uniform(1, 9) * severity_factor,
                "magnitude_scale": np.random.choice(['Richter', 'Saffir-Simpson', 'Fujita', 'Meters', 'km²']),
                "deaths": int(np.random.exponential(100) * severity_factor),
                "injured": int(np.random.exponential(500) * severity_factor),
                "affected": int(np.random.exponential(10000) * severity_factor),
                "homeless": int(np.random.exponential(5000) * severity_factor),
                "total_affected": int(np.random.exponential(20000) * severity_factor),
                "total_damages": int(np.random.exponential(50000000) * severity_factor) / 1000000,  # in millions
                "total_damages_adjusted": int(np.random.exponential(100000000) * severity_factor) / 1000000,  # in millions
            }
            
            data.append(record)
            disaster_id += 1
    
    logging.info(f"Generated {len(data)} disaster records")
    return data

if __name__ == "__main__":
    fetch_emdat_data()

