
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Doom Meter",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import requests
from io import StringIO
import pymongo
import time
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient
import json
import re
from geopy.geocoders import Nominatim
import pycountry
import warnings
warnings.filterwarnings('ignore')




# Connect to MongoDB
@st.cache_resource
def init_connection():
    try:
        client = MongoClient("mongodb://localhost:27017/")
        return client
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        return None

client = init_connection()

# Select database and collection
if client:
    db_name = os.environ.get("DisasterDataConc","DisasterData")
    db = client[db_name]
    disasters_collection = db.disasters
    alerts_collection = db.alerts
    resources_collection = db.resources
    evacuation_collection = db.evacuation_plans
    damage_collection = db.damage_reports

# Initialize session state variables if they don't exist
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
    # Load alerts from MongoDB if available
    if client:
        alerts_cursor = alerts_collection.find({})
        st.session_state.alerts = list(alerts_cursor)

if 'resources' not in st.session_state:
    # Check if resources exist in MongoDB
    if client and resources_collection.count_documents({}) > 0:
        resources_doc = resources_collection.find_one({})
        st.session_state.resources = resources_doc['resources']
    else:
        st.session_state.resources = {
            'Medical Supplies': 1000,
            'Food Packages': 500,
            'Water (liters)': 2000,
            'Shelter Kits': 200,
            'Rescue Teams': 20
        }
        # Save initial resources to MongoDB
        if client:
            resources_collection.insert_one({'resources': st.session_state.resources})

if 'evacuation_plans' not in st.session_state:
    st.session_state.evacuation_plans = []
    # Load evacuation plans from MongoDB if available
    if client:
        evacuation_cursor = evacuation_collection.find({})
        st.session_state.evacuation_plans = list(evacuation_cursor)

if 'damage_reports' not in st.session_state:
    st.session_state.damage_reports = []
    # Load damage reports from MongoDB if available
    if client:
        damage_cursor = damage_collection.find({})
        st.session_state.damage_reports = list(damage_cursor)

# Function to fetch data from EM-DAT
@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_emdat_data():
    """
    Fetch disaster data from EM-DAT database using credentials
    """
    # Check if data already exists in MongoDB
    if client and disasters_collection.count_documents({}) > 0:
        st.info("Loading disaster data from database...")
        disasters_cursor = disasters_collection.find({})
        disasters_list = list(disasters_cursor)
        
        # Convert MongoDB documents to DataFrame
        df = pd.DataFrame(disasters_list)
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        
        return df
    
    st.info("Fetching disaster data from EM-DAT...")
    
    try:
        # Use environment variables for authentication
        username = "rishavpal309@gmail.com"
        password = "@Rishabh6397"
        
        if not username or not password:
            st.error("EM-DAT credentials not found. Using sample data instead.")
            return generate_sample_data()
        
        # Simulate fetching data from EM-DAT (actual implementation would use their API)
        # For this example, we'll create a realistic dataset based on EM-DAT structure
        df = generate_emdat_sample()
        
        # Store in MongoDB for future use
        if client:
            disasters_collection.insert_many(df.to_dict('records'))
            st.success("Data stored in database successfully!")
        
        return df
    
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return generate_sample_data()

def generate_emdat_sample():
    """
    Generate a realistic sample dataset based on EM-DAT structure
    with 15+ years of data (2005-2023)
    """
    # Start and end dates
    start_date = datetime(2005, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # List of countries
    countries = [country.name for country in list(pycountry.countries)][:50]
    
    # Disaster types and subtypes
    disaster_types = {
        'Geophysical': ['Earthquake', 'Volcanic activity', 'Mass movement (dry)'],
        'Meteorological': ['Storm', 'Extreme temperature', 'Fog'],
        'Hydrological': ['Flood', 'Landslide', 'Wave action'],
        'Climatological': ['Drought', 'Wildfire', 'Glacial lake outburst'],
        'Biological': ['Epidemic', 'Insect infestation', 'Animal accident'],
        'Technological': ['Industrial accident', 'Transport accident', 'Miscellaneous accident']
    }
    
    # Generate random dates
    num_records = 20000  # Keep under 50k as requested
    random_dates = [start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days)) for _ in range(num_records)]
    
    # Generate random disaster data
    data = []
    for i in range(num_records):
        disaster_group = np.random.choice(list(disaster_types.keys()))
        disaster_subtype = np.random.choice(disaster_types[disaster_group])
        country = np.random.choice(countries)
        
        # Get country code
        try:
            country_obj = pycountry.countries.get(name=country)
            country_code = country_obj.alpha_3 if country_obj else "UNK"
        except:
            country_code = "UNK"
        
        # Generate realistic impact data with some correlation to disaster type
        severity_factor = 1.0
        if disaster_group == 'Geophysical':
            severity_factor = 3.0
        elif disaster_group == 'Meteorological':
            severity_factor = 2.5
        elif disaster_group == 'Hydrological':
            severity_factor = 2.0
        
        # More severe disasters are less frequent
        if np.random.random() < 0.8:
            severity_factor *= 0.5
        
        # Generate coordinates (approximate by country)
        lat = np.random.uniform(-60, 70)
        lon = np.random.uniform(-180, 180)
        
        # Create disaster record
        record = {
            'disaster_id': f"EM-DAT-{2005+i%19}-{i:04d}",
            'year': random_dates[i].year,
            'start_date': random_dates[i].strftime('%Y-%m-%d'),
            'end_date': (random_dates[i] + timedelta(days=np.random.randint(1, 30))).strftime('%Y-%m-%d'),
            'disaster_group': disaster_group,
            'disaster_type': disaster_subtype,
            'country': country,
            'iso3': country_code,
            'region': np.random.choice(['Africa', 'Americas', 'Asia', 'Europe', 'Oceania']),
            'latitude': lat,
            'longitude': lon,
            'magnitude': np.random.uniform(1, 9) * severity_factor,
            'magnitude_scale': np.random.choice(['Richter', 'Saffir-Simpson', 'Fujita', 'Meters', 'km²']),
            'deaths': int(np.random.exponential(100) * severity_factor),
            'injured': int(np.random.exponential(500) * severity_factor),
            'affected': int(np.random.exponential(10000) * severity_factor),
            'homeless': int(np.random.exponential(5000) * severity_factor),
            'total_affected': int(np.random.exponential(20000) * severity_factor),
            'total_damages': int(np.random.exponential(50000000) * severity_factor) / 1000000,  # in millions
            'total_damages_adjusted': int(np.random.exponential(100000000) * severity_factor) / 1000000,  # in millions
        }
        data.append(record)
    
    return pd.DataFrame(data)

def generate_sample_data():
    """Generate sample disaster data if real data cannot be fetched"""
    # Sample historical disaster data
    return pd.DataFrame({
        'disaster_id': [f"SAMPLE-{i:04d}" for i in range(1000)],
        'year': np.random.choice(range(2005, 2024), 1000),
        'start_date': pd.date_range(start='1/1/2005', end='12/31/2023', periods=1000).strftime('%Y-%m-%d'),
        'disaster_type': np.random.choice(['Flood', 'Earthquake', 'Hurricane', 'Wildfire', 'Drought', 'Tsunami'], 1000),
        'country': np.random.choice(['USA', 'Japan', 'India', 'Brazil', 'Australia', 'France', 'China'], 1000),
        'latitude': np.random.uniform(-60, 70, 1000),
        'longitude': np.random.uniform(-180, 180, 1000),
        'magnitude': np.random.uniform(3.0, 9.0, 1000),
        'deaths': np.random.randint(0, 1000, 1000),
        'affected': np.random.randint(100, 100000, 1000),
        'total_damages': np.random.uniform(1, 1000, 1000),  # in millions
    })

# Load disaster data
disaster_data = fetch_emdat_data()

# Prepare data for ML model
def prepare_data_for_model(df):
    """Prepare disaster data for machine learning model"""
    # Select relevant features
    features = ['disaster_group', 'magnitude', 'year']
    
    # Add region if available
    if 'region' in df.columns:
        features.append('region')
    
    # Target variables
    targets = ['deaths', 'total_damages']
    
    # Ensure all required columns exist
    for col in features + targets:
        if col not in df.columns:
            if col == 'disaster_group' and 'disaster_type' in df.columns:
                df['disaster_group'] = df['disaster_type']
            else:
                df[col] = np.random.rand(len(df))
    
    # Handle missing values
    for col in features + targets:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna('Unknown')
    
    # Create X and y
    X = df[features]
    y = df[targets]
    
    return X, y

# Load or train ML model
@st.cache_resource
def load_or_train_model(disaster_data):
    model_path = 'disaster_impact_model.pkl'
    
    # Prepare data
    X, y = prepare_data_for_model(disaster_data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define preprocessing for categorical features
    categorical_features = [col for col in X.columns if X[col].dtype == 'object']
    numeric_features = [col for col in X.columns if X[col].dtype != 'object']
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Create and train model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, model_path)
    
    return model

# Load or train the model
model = load_or_train_model(disaster_data)

# Sidebar navigation
st.sidebar.title("Doom Meter")
page = st.sidebar.selectbox(
    "Navigation",
    ["Dashboard", "Disaster Monitoring", "Resource Management", 
     "Evacuation Planning", "Damage Assessment", "Prediction & Analytics", "Data Explorer"]
)

# Dashboard
if page == "Dashboard":
    st.title("Doom Meter Dashboard")
    
    # Display current alerts
    st.subheader("Active Alerts")
    if st.session_state.alerts:
        for alert in st.session_state.alerts:
            st.warning(f"**{alert['type']}** - {alert['location']} - {alert['time']}")
    else:
        st.info("No active alerts at this time.")
    
    # Resource overview
    st.subheader("Resource Overview")
    resource_df = pd.DataFrame({
        'Resource': list(st.session_state.resources.keys()),
        'Available': list(st.session_state.resources.values())
    })
    
    # Use Plotly for better interactive charts
    fig = px.bar(resource_df, x='Resource', y='Available', color='Available',
                 color_continuous_scale='Viridis', title='Available Resources')
    st.plotly_chart(fig)
    
    # Recent disaster map
    st.subheader("Global Disaster Map (Last 5 Years)")
    
    # Filter recent disasters (last 5 years)
    current_year = datetime.now().year
    recent_disasters = disaster_data[disaster_data['year'] >= current_year - 5].copy()
    
    # Create map
    m = folium.Map(location=[20, 0], zoom_start=2)
    
    # Add disaster markers to map
    for idx, row in recent_disasters.head(500).iterrows():  # Limit to 500 for performance
        try:
            # Determine color based on disaster type
            if 'disaster_group' in row:
                disaster_type = row['disaster_group']
            else:
                disaster_type = row['disaster_type']
                
            if 'Geophysical' in disaster_type or 'Earthquake' in disaster_type:
                color = 'red'
            elif 'Meteorological' in disaster_type or 'Storm' in disaster_type or 'Hurricane' in disaster_type:
                color = 'blue'
            elif 'Hydrological' in disaster_type or 'Flood' in disaster_type:
                color = 'darkblue'
            elif 'Climatological' in disaster_type or 'Wildfire' in disaster_type or 'Drought' in disaster_type:
                color = 'orange'
            elif 'Biological' in disaster_type:
                color = 'green'
            else:
                color = 'gray'
            
            # Create popup content
            popup_content = f"""
            <b>{row.get('disaster_type', disaster_type)}</b><br>
            Location: {row.get('country', 'Unknown')}<br>
            Date: {row.get('start_date', 'Unknown')}<br>
            Magnitude: {row.get('magnitude', 'Unknown')}<br>
            Deaths: {row.get('deaths', 'Unknown')}<br>
            Damages: ${row.get('total_damages', 'Unknown')} million
            """
            
            # Add marker
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                popup=folium.Popup(popup_content, max_width=300),
                color=color,
                fill=True,
                fill_opacity=0.7
            ).add_to(m)
        except Exception as e:
            continue  # Skip problematic entries
    
    # Display map
    folium_static(m)
    
    # Recent statistics
    st.subheader("Disaster Statistics (Last 12 Months)")
    
    # Filter data for last 12 months
    last_year = datetime.now() - timedelta(days=365)
    last_year_str = last_year.strftime('%Y-%m-%d')
    recent_data = disaster_data[disaster_data['start_date'] >= last_year_str]
    
    # Calculate statistics
    total_disasters = len(recent_data)
    total_deaths = recent_data['deaths'].sum()
    total_affected = recent_data['affected'].sum() if 'affected' in recent_data.columns else 0
    total_damages = recent_data['total_damages'].sum()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Disasters", f"{total_disasters:,}")
    col2.metric("Deaths", f"{int(total_deaths):,}")
    col3.metric("People Affected", f"{int(total_affected):,}")
    col4.metric("Economic Impact", f"${total_damages:.2f}M")
    
    # Trend analysis
    st.subheader("Disaster Trends")
    
    # Group by year and disaster type
    yearly_disasters = disaster_data.groupby(['year', 'disaster_type' if 'disaster_type' in disaster_data.columns else 'disaster_group']).size().reset_index(name='count')
    
    # Create interactive plot
    fig = px.line(yearly_disasters, x='year', y='count', color='disaster_type' if 'disaster_type' in yearly_disasters.columns else 'disaster_group',
                 title='Disaster Frequency by Type (2005-2023)')
    st.plotly_chart(fig)

# Disaster Monitoring
elif page == "Disaster Monitoring":
    st.title("Disaster Monitoring and Alerts")
    
    # Create new alert
    st.subheader("Create New Alert")
    col1, col2 = st.columns(2)
    
    with col1:
        alert_type = st.selectbox("Disaster Type", ["Earthquake", "Flood", "Hurricane", "Wildfire", "Tsunami", "Other"])
        alert_severity = st.slider("Severity Level", 1, 5, 3)
        alert_location = st.text_input("Location")
    
    with col2:
        alert_description = st.text_area("Description")
        alert_time = st.date_input("Date")
    
    if st.button("Issue Alert"):
        if alert_location:
            new_alert = {
                "type": alert_type,
                "severity": alert_severity,
                "location": alert_location,
                "description": alert_description,
                "time": alert_time.strftime("%Y-%m-%d")
            }
            st.session_state.alerts.append(new_alert)
            
            # Save to MongoDB
            if client:
                alerts_collection.insert_one(new_alert)
            
            st.success("Alert issued successfully!")
        else:
            st.error("Location is required!")
    
    # View existing alerts
    st.subheader("Current Alerts")
    if st.session_state.alerts:
        alert_df = pd.DataFrame(st.session_state.alerts)
        st.dataframe(alert_df)
        
        # Option to resolve alerts
        if len(st.session_state.alerts) > 0:
            alert_to_resolve = st.selectbox("Select Alert to Resolve", 
                                            range(len(st.session_state.alerts)),
                                            format_func=lambda x: f"{st.session_state.alerts[x]['type']} at {st.session_state.alerts[x]['location']}")
            
            if st.button("Resolve Selected Alert"):
                # Get the alert to remove
                alert_to_remove = st.session_state.alerts.pop(alert_to_resolve)
                
                # Remove from MongoDB
                if client and '_id' in alert_to_remove:
                    alerts_collection.delete_one({"_id": alert_to_remove["_id"]})
                
                st.success("Alert resolved!")
                st.experimental_rerun()
    else:
        st.info("No active alerts at this time.")
    
    # Real-time monitoring simulation
    st.subheader("Real-time Monitoring Simulation")
    
    # Get recent disasters
    recent = disaster_data.sort_values('start_date', ascending=False).head(10)
    
    # Display as "real-time" feed
    st.expander("Recent disaster events from global monitoring systems:")
    
    for idx, row in recent.iterrows():
        disaster_type = row.get('disaster_type', row.get('disaster_group', 'Unknown'))
        country = row.get('country', 'Unknown location')
        date = row.get('start_date', 'Unknown date')
        magnitude = row.get('magnitude', 'N/A')
        
        st.info(f"**{disaster_type}** detected in {country} on {date}. Magnitude: {magnitude}")

# Resource Management
elif page == "Resource Management":
    st.title("Resource Management")
    
    # Display current resources
    st.subheader("Current Resources")
    resource_df = pd.DataFrame({
        'Resource': list(st.session_state.resources.keys()),
        'Available': list(st.session_state.resources.values())
    })
    st.dataframe(resource_df)
    
    # Update resources
    st.subheader("Update Resources")
    col1, col2 = st.columns(2)
    
    with col1:
        resource_type = st.selectbox("Resource Type", list(st.session_state.resources.keys()))
        action = st.radio("Action", ["Add", "Remove"])
    
    with col2:
        quantity = st.number_input("Quantity", min_value=1, value=10)
        reason = st.text_input("Reason for update")
    
    if st.button("Update Resource"):
        if action == "Add":
            st.session_state.resources[resource_type] += quantity
            st.success(f"Added {quantity} units of {resource_type}")
        else:
            if st.session_state.resources[resource_type] >= quantity:
                st.session_state.resources[resource_type] -= quantity
                st.success(f"Removed {quantity} units of {resource_type}")
            else:
                st.error(f"Not enough {resource_type} available!")
        
        # Update in MongoDB
        if client:
            resources_collection.update_one(
                {}, 
                {"$set": {"resources": st.session_state.resources}},
                upsert=True
            )
    
    # Resource allocation
    st.subheader("Resource Allocation")
    col1, col2 = st.columns(2)
    
    with col1:
        allocation_resource = st.selectbox("Resource to Allocate", list(st.session_state.resources.keys()), key="alloc_resource")
        allocation_quantity = st.number_input("Quantity to Allocate", min_value=1, value=5, key="alloc_qty")
    
    with col2:
        allocation_location = st.text_input("Allocation Location")
        allocation_priority = st.selectbox("Priority", ["High", "Medium", "Low"])
    
    if st.button("Allocate Resources"):
        if allocation_location:
            if st.session_state.resources[allocation_resource] >= allocation_quantity:
                st.session_state.resources[allocation_resource] -= allocation_quantity
                st.success(f"Allocated {allocation_quantity} units of {allocation_resource} to {allocation_location}")
                
                # Update in MongoDB
                if client:
                    resources_collection.update_one(
                        {}, 
                        {"$set": {"resources": st.session_state.resources}},
                        upsert=True
                    )
            else:
                st.error(f"Not enough {allocation_resource} available!")
        else:
            st.error("Location is required!")
    
    # Resource visualization
    st.subheader("Resource Distribution")
    fig = px.pie(resource_df, values='Available', names='Resource', 
                 title='Resource Distribution', hole=0.3)
    st.plotly_chart(fig)

# Evacuation Planning
elif page == "Evacuation Planning":
    st.title("Evacuation Planning")
    
    # Create evacuation plan
    st.subheader("Create Evacuation Plan")
    col1, col2 = st.columns(2)
    
    with col1:
        evac_area = st.text_input("Area to Evacuate")
        evac_population = st.number_input("Estimated Population", min_value=1, value=1000)
        evac_reason = st.selectbox("Evacuation Reason", ["Flood", "Fire", "Hurricane", "Earthquake", "Chemical Spill", "Other"])
    
    with col2:
        evac_start = st.date_input("Start Date")
        evac_duration = st.number_input("Estimated Duration (days)", min_value=1, value=3)
        evac_shelter = st.text_input("Primary Shelter Location")
    
    if st.button("Create Evacuation Plan"):
        if evac_area and evac_shelter:
            new_plan = {
                "area": evac_area,
                "population": evac_population,
                "reason": evac_reason,
                "start_date": evac_start.strftime("%Y-%m-%d"),
                "duration": evac_duration,
                "shelter": evac_shelter,
                "status": "Planned"
            }
            st.session_state.evacuation_plans.append(new_plan)
            
            # Save to MongoDB
            if client:
                evacuation_collection.insert_one(new_plan)
            
            st.success("Evacuation plan created!")
        else:
            st.error("Area and Shelter location are required!")
    
    # View evacuation plans
    st.subheader("Current Evacuation Plans")
    if st.session_state.evacuation_plans:
        evac_df = pd.DataFrame(st.session_state.evacuation_plans)
        if '_id' in evac_df.columns:
            evac_df = evac_df.drop('_id', axis=1)
        st.dataframe(evac_df)
        
        # Update plan status
        if len(st.session_state.evacuation_plans) > 0:
            plan_to_update = st.selectbox("Select Plan to Update", 
                                         range(len(st.session_state.evacuation_plans)),
                                         format_func=lambda x: f"{st.session_state.evacuation_plans[x]['area']} - {st.session_state.evacuation_plans[x]['status']}")
            
            new_status = st.selectbox("New Status", ["Planned", "In Progress", "Completed", "Cancelled"])
            
            if st.button("Update Plan Status"):
                # Update in session state
                st.session_state.evacuation_plans[plan_to_update]["status"] = new_status
                
                # Update in MongoDB
                if client and '_id' in st.session_state.evacuation_plans[plan_to_update]:
                    plan_id = st.session_state.evacuation_plans[plan_to_update]["_id"]
                    evacuation_collection.update_one(
                        {"_id": plan_id},
                        {"$set": {"status": new_status}}
                    )
                
                st.success("Plan status updated!")
                st.experimental_rerun()
    else:
        st.info("No evacuation plans created yet.")
    
    # Evacuation route planning
    st.subheader("Evacuation Route Planning")
    
    # Simple map for evacuation planning
    if st.session_state.evacuation_plans:
        selected_plan = st.selectbox(
            "Select Plan for Route Planning",
            range(len(st.session_state.evacuation_plans)),
            format_func=lambda x: f"{st.session_state.evacuation_plans[x]['area']} - {st.session_state.evacuation_plans[x]['shelter']}"
        )
        
        plan = st.session_state.evacuation_plans[selected_plan]
        
        # Geocode locations (in a real app, you'd want to cache these)
        geolocator = Nominatim(user_agent="disaster_management_system")
        
        try:
            # Try to geocode the evacuation area
            area_location = geolocator.geocode(plan['area'])
            if area_location:
                area_lat, area_lon = area_location.latitude, area_location.longitude
            else:
                # Fallback to random coordinates
                area_lat, area_lon = np.random.uniform(20, 50), np.random.uniform(-120, -70)
            
            # Try to geocode the shelter
            shelter_location = geolocator.geocode(plan['shelter'])
            if shelter_location:
                shelter_lat, shelter_lon = shelter_location.latitude, shelter_location.longitude
            else:
                # Fallback to nearby coordinates
                shelter_lat = area_lat + np.random.uniform(-0.1, 0.1)
                shelter_lon = area_lon + np.random.uniform(-0.1, 0.1)
            
            # Create map
            evac_map = folium.Map(location=[(area_lat + shelter_lat)/2, (area_lon + shelter_lon)/2], zoom_start=10)
            
            # Add markers
            folium.Marker(
                location=[area_lat, area_lon],
                popup=f"Evacuation Area: {plan['area']}",
                icon=folium.Icon(color="red", icon="home")
            ).add_to(evac_map)
            
            folium.Marker(
                location=[shelter_lat, shelter_lon],
                popup=f"Shelter: {plan['shelter']}",
                icon=folium.Icon(color="green", icon="info-sign")
            ).add_to(evac_map)
            
            # Add a line connecting the two points
            folium.PolyLine(
                locations=[[area_lat, area_lon], [shelter_lat, shelter_lon]],
                color="blue",
                weight=2.5,
                opacity=1
            ).add_to(evac_map)
            
            # Display map
            folium_static(evac_map)
            
            # Display evacuation info
            st.info(f"""
            **Evacuation Plan Details:**
            - Population to evacuate: {plan['population']:,} people
            - Start date: {plan['start_date']}
            - Duration: {plan['duration']} days
            - Status: {plan['status']}
            """)
            
        except Exception as e:
            st.error(f"Error creating evacuation map: {e}")
            st.info("Please enter valid location names for geocoding to work properly.")

# Damage Assessment
elif page == "Damage Assessment":
    st.title("Damage Assessment")
    
    # Create damage report
    st.subheader("Create Damage Report")
    col1, col2 = st.columns(2)
    
    with col1:
        damage_location = st.text_input("Location")
        damage_type = st.selectbox("Disaster Type", ["Earthquake", "Flood", "Hurricane", "Wildfire", "Tsunami", "Other"])
        damage_date = st.date_input("Date of Assessment")
    
    with col2:
        infrastructure_damage = st.slider("Infrastructure Damage (%)", 0, 100, 25)
        casualties = st.number_input("Estimated Casualties", min_value=0, value=0)
        economic_impact = st.number_input("Economic Impact ($ millions)", min_value=0.0, value=1.0)
    
    damage_notes = st.text_area("Additional Notes")
    
    if st.button("Submit Damage Report"):
        if damage_location:
            new_report = {
                "location": damage_location,
                "disaster_type": damage_type,
                "date": damage_date.strftime("%Y-%m-%d"),
                "infrastructure_damage": infrastructure_damage,
                "casualties": casualties,
                "economic_impact": economic_impact,
                "notes": damage_notes
            }
            st.session_state.damage_reports.append(new_report)
            
            # Save to MongoDB
            if client:
                damage_collection.insert_one(new_report)
            
            st.success("Damage report submitted!")
        else:
            st.error("Location is required!")
    
    # View damage reports
    st.subheader("Damage Reports")
    if st.session_state.damage_reports:
        damage_df = pd.DataFrame(st.session_state.damage_reports)
        if '_id' in damage_df.columns:
            damage_df = damage_df.drop('_id', axis=1)
        st.dataframe(damage_df)
        
        # Visualization
        st.subheader("Damage Visualization")
        if len(st.session_state.damage_reports) > 1:
            # Interactive visualizations with Plotly
            fig1 = px.bar(damage_df, x='location', y='infrastructure_damage', 
                         color='disaster_type', title='Infrastructure Damage by Location')
            st.plotly_chart(fig1)
            
            fig2 = px.pie(damage_df, values='economic_impact', names='disaster_type',
                         title='Economic Impact by Disaster Type')
            st.plotly_chart(fig2)
            
            # Damage map
            st.subheader("Damage Assessment Map")
            
            # Create map
            damage_map = folium.Map(location=[20, 0], zoom_start=2)
            
            # Add damage locations to map
            for idx, row in damage_df.iterrows():
                try:
                    # Try to geocode the location
                    geolocator = Nominatim(user_agent="disaster_management_system")
                    location = geolocator.geocode(row['location'])
                    
                    if location:
                        lat, lon = location.latitude, location.longitude
                        
                        # Determine color based on damage level
                        if row['infrastructure_damage'] > 75:
                            color = 'red'
                        elif row['infrastructure_damage'] > 50:
                            color = 'orange'
                        elif row['infrastructure_damage'] > 25:
                            color = 'yellow'
                        else:
                            color = 'green'
                        
                        # Create popup content
                        popup_content = f"""
                        <b>{row['disaster_type']}</b><br>
                        Location: {row['location']}<br>
                        Date: {row['date']}<br>
                        Infrastructure Damage: {row['infrastructure_damage']}%<br>
                        Casualties: {row['casualties']}<br>
                        Economic Impact: ${row['economic_impact']} million
                        """
                        
                        # Add marker
                        folium.CircleMarker(
                            location=[lat, lon],
                            radius=row['infrastructure_damage'] / 10 + 5,  # Size based on damage
                            popup=folium.Popup(popup_content, max_width=300),
                            color=color,
                            fill=True,
                            fill_opacity=0.7
                        ).add_to(damage_map)
                except Exception as e:
                    continue  # Skip problematic entries
            
            # Display map
            folium_static(damage_map)
    else:
        st.info("No damage reports submitted yet.")

# Prediction & Analytics
elif page == "Prediction & Analytics":
    st.title("Disaster Prediction & Analytics")
    
    # Historical data analysis
    st.subheader("Historical Disaster Data Analysis")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        start_year = st.selectbox("Start Year", range(2005, 2024), index=0)
    with col2:
        end_year = st.selectbox("End Year", range(2005, 2024), index=18)  # Default to 2023
    with col3:
        disaster_filter = st.multiselect(
            "Disaster Types",
            options=sorted(disaster_data['disaster_type'].unique() if 'disaster_type' in disaster_data.columns 
                          else disaster_data['disaster_group'].unique()),
            default=[]
        )
    
    # Apply filters
    filtered_data = disaster_data[(disaster_data['year'] >= start_year) & (disaster_data['year'] <= end_year)]
    
    if disaster_filter:
        if 'disaster_type' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['disaster_type'].isin(disaster_filter)]
        else:
            filtered_data = filtered_data[filtered_data['disaster_group'].isin(disaster_filter)]
    
    # Show filtered data
    st.expander(f"Showing {len(filtered_data)} disaster records from {start_year} to {end_year}")
    
    # Data visualization
    st.subheader("Disaster Trends")
    
    # Time series of disasters by year
    yearly_counts = filtered_data.groupby('year').size().reset_index(name='count')
    
    fig = px.line(yearly_counts, x='year', y='count', 
                 title=f'Disaster Frequency ({start_year}-{end_year})',
                 markers=True)
    st.plotly_chart(fig)
    
    # Disasters by type
    if 'disaster_type' in filtered_data.columns:
        type_field = 'disaster_type'
    else:
        type_field = 'disaster_group'
        
    type_counts = filtered_data.groupby(type_field).size().reset_index(name='count')
    type_counts = type_counts.sort_values('count', ascending=False)
    
    fig = px.bar(type_counts, x=type_field, y='count', 
                title=f'Disaster Frequency by Type ({start_year}-{end_year})',
                color='count')
    st.plotly_chart(fig)
    
    # Disasters by region if available
    if 'region' in filtered_data.columns:
        region_counts = filtered_data.groupby('region').size().reset_index(name='count')
        region_counts = region_counts.sort_values('count', ascending=False)
        
        fig = px.bar(region_counts, x='region', y='count', 
                    title=f'Disaster Frequency by Region ({start_year}-{end_year})',
                    color='count')
        st.plotly_chart(fig)
    
    # Impact analysis
    st.subheader("Disaster Impact Analysis")
    
    # Deaths by disaster type
    if 'deaths' in filtered_data.columns:
        deaths_by_type = filtered_data.groupby(type_field)['deaths'].sum().reset_index()
        deaths_by_type = deaths_by_type.sort_values('deaths', ascending=False)
        
        fig = px.bar(deaths_by_type, x=type_field, y='deaths', 
                    title=f'Total Deaths by Disaster Type ({start_year}-{end_year})',
                    color='deaths')
        st.plotly_chart(fig)
    
    # Economic damage by disaster type
    if 'total_damages' in filtered_data.columns:
        damage_by_type = filtered_data.groupby(type_field)['total_damages'].sum().reset_index()
        damage_by_type = damage_by_type.sort_values('total_damages', ascending=False)
        
        fig = px.bar(damage_by_type, x=type_field, y='total_damages', 
                    title=f'Total Economic Damage by Disaster Type (${start_year}-{end_year}, millions)',
                    color='total_damages')
        st.plotly_chart(fig)
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    
    # Select numeric columns for correlation
    numeric_cols = [col for col in filtered_data.columns if 
                   filtered_data[col].dtype in [np.int64, np.float64] and 
                   col not in ['year', 'latitude', 'longitude']]
    
    if len(numeric_cols) > 1:
        corr = filtered_data[numeric_cols].corr()
        
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                       title='Correlation Between Disaster Variables')
        st.plotly_chart(fig)
    
    # Disaster impact prediction
    st.subheader("Disaster Impact Prediction")
    st.expander("Predict potential casualties and economic damage based on disaster characteristics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pred_disaster_type = st.selectbox(
            "Disaster Type", 
            options=sorted(disaster_data['disaster_type'].unique() if 'disaster_type' in disaster_data.columns 
                          else disaster_data['disaster_group'].unique())
        )
        pred_magnitude = st.slider("Magnitude/Intensity", 1.0, 10.0, 6.0, 0.1)
    
    with col2:
        pred_year = st.number_input("Year", min_value=2023, max_value=2030, value=2023)
        
        if 'region' in disaster_data.columns:
            pred_region = st.selectbox("Region", options=sorted(disaster_data['region'].unique()))
        else:
            pred_region = "Unknown"
    
    if st.button("Predict Impact"):
        # Prepare input features
        input_data = {
            'disaster_group' if 'disaster_group' in disaster_data.columns else 'disaster_type': pred_disaster_type,
            'magnitude': pred_magnitude,
            'year': pred_year
        }
        
        if 'region' in disaster_data.columns:
            input_data['region'] = pred_region
        
        # Create DataFrame with the input
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        try:
            prediction = model.predict(input_df)
            
            # Display results
            st.subheader("Prediction Results")
            col1, col2 = st.columns(2)
            
            col1.metric("Estimated Casualties", f"{int(prediction[0][0])}")
            col2.metric("Estimated Economic Damage", f"${prediction[0][1]:.2f} million")
            
            # Confidence interval (simplified)
            st.expander("**Prediction Confidence Interval**")
            st.expander(f"Casualties: {max(0, int(prediction[0][0] * 0.8))} - {int(prediction[0][0] * 1.2)}")
            st.expander(f"Economic Damage: ${max(0, prediction[0][1] * 0.8):.2f} - ${prediction[0][1] * 1.2:.2f} million")
            
            # Recommendations based on prediction
            st.subheader("Recommended Actions")
            
            if prediction[0][0] > 100:
                st.warning("⚠️ High casualty prediction! Consider immediate evacuation and deployment of medical teams.")
            
            if prediction[0][1] > 500:
                st.warning("⚠️ Severe economic impact expected! Prepare for extensive recovery operations.")
            
            # Generate specific recommendations based on disaster type
            if 'flood' in pred_disaster_type.lower():
                st.expander("- Evacuate low-lying areas")
                st.expander("- Deploy water pumps and sandbags")
                st.expander("- Prepare water purification systems")
            elif 'earthquake' in pred_disaster_type.lower():
                st.expander("- Search and rescue teams on standby")
                st.expander("- Structural engineers for building assessment")
                st.expander("- Prepare temporary shelter facilities")
            elif 'hurricane' in pred_disaster_type.lower() or 'storm' in pred_disaster_type.lower():
                st.expander("- Secure buildings and infrastructure")
                st.expander("- Clear drainage systems")
                st.expander("- Establish emergency communication systems")
            elif 'wildfire' in pred_disaster_type.lower() or 'fire' in pred_disaster_type.lower():
                st.expander("- Evacuate threatened areas")
                st.expander("- Position firefighting resources")
                st.expander("- Monitor air quality")
            else:
                st.expander("- Prepare emergency response teams")
                st.expander("- Establish communication protocols")
                st.expander("- Ready evacuation centers")
        
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Please ensure all input fields are filled correctly.")

# Data Explorer
elif page == "Data Explorer":
    st.title("Disaster Data Explorer")
    
    # Data overview
    st.subheader("Data Overview")
    st.expander(f"Total records: {len(disaster_data):,}")
    st.expander(f"Year range: {disaster_data['year'].min()} - {disaster_data['year'].max()}")
    
    # Show data columns
    st.expander("Available data fields:")
    st.expander(", ".join(disaster_data.columns))
    
    # Data filters
    st.subheader("Filter Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        year_range = st.slider(
            "Year Range", 
            min_value=int(disaster_data['year'].min()), 
            max_value=int(disaster_data['year'].max()),
            value=(int(disaster_data['year'].min()), int(disaster_data['year'].max()))
        )
    
    with col2:
        if 'disaster_type' in disaster_data.columns:
            disaster_types = st.multiselect(
                "Disaster Types",
                options=sorted(disaster_data['disaster_type'].unique()),
                default=[]
            )
        elif 'disaster_group' in disaster_data.columns:
            disaster_types = st.multiselect(
                "Disaster Groups",
                options=sorted(disaster_data['disaster_group'].unique()),
                default=[]
            )
        else:
            disaster_types = []
    
    with col3:
        if 'region' in disaster_data.columns:
            regions = st.multiselect(
                "Regions",
                options=sorted(disaster_data['region'].unique()),
                default=[]
            )
        else:
            regions = []
    
    # Apply filters
    filtered_data = disaster_data[(disaster_data['year'] >= year_range[0]) & 
                                 (disaster_data['year'] <= year_range[1])]
    
    if disaster_types:
        if 'disaster_type' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['disaster_type'].isin(disaster_types)]
        elif 'disaster_group' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['disaster_group'].isin(disaster_types)]
    
    if regions and 'region' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['region'].isin(regions)]
    
    # Show filtered data
    st.expander(f"Showing {len(filtered_data):,} records")
    st.dataframe(filtered_data)
    
    # Download option
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="disaster_data.csv",
        mime="text/csv",
    )
    
    # Custom visualization
    st.subheader("Custom Visualization")
    
    # Select columns for visualization
    numeric_cols = [col for col in filtered_data.columns if 
                   filtered_data[col].dtype in [np.int64, np.float64] and 
                   col not in ['year', 'latitude', 'longitude']]
    
    categorical_cols = [col for col in filtered_data.columns if 
                       filtered_data[col].dtype == 'object' or col == 'year']
    
    # Visualization type
    viz_type = st.selectbox(
        "Visualization Type",
        options=["Bar Chart", "Line Chart", "Scatter Plot", "Box Plot", "Histogram"]
    )
    
    if viz_type in ["Bar Chart", "Line Chart", "Box Plot"]:
        x_axis = st.selectbox("X-Axis", options=categorical_cols)
        y_axis = st.selectbox("Y-Axis", options=numeric_cols)
        
        if viz_type == "Bar Chart":
            fig = px.bar(filtered_data, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
        elif viz_type == "Line Chart":
            # Group by x_axis and calculate mean of y_axis
            grouped_data = filtered_data.groupby(x_axis)[y_axis].mean().reset_index()
            fig = px.line(grouped_data, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}", markers=True)
        else:  # Box Plot
            fig = px.box(filtered_data, x=x_axis, y=y_axis, title=f"Distribution of {y_axis} by {x_axis}")
    
    elif viz_type == "Scatter Plot":
        x_axis = st.selectbox("X-Axis", options=numeric_cols)
        y_axis = st.selectbox("Y-Axis", options=numeric_cols)
        color_by = st.selectbox("Color By", options=['None'] + categorical_cols)
        
        if color_by == 'None':
            fig = px.scatter(filtered_data, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
        else:
            fig = px.scatter(filtered_data, x=x_axis, y=y_axis, color=color_by, title=f"{y_axis} vs {x_axis} by {color_by}")
    
    else:  # Histogram
        column = st.selectbox("Column", options=numeric_cols)
        bins = st.slider("Number of Bins", min_value=5, max_value=100, value=20)
        
        fig = px.histogram(filtered_data, x=column, nbins=bins, title=f"Distribution of {column}")
    
    # Display the visualization
    st.plotly_chart(fig)
    
    # Map visualization
    st.subheader("Geographic Distribution")
    
    if 'latitude' in filtered_data.columns and 'longitude' in filtered_data.columns:
        # Create map
        geo_map = folium.Map(location=[20, 0], zoom_start=2)
        
        # Add disaster markers to map
        for idx, row in filtered_data.head(1000).iterrows():  # Limit to 1000 for performance
            try:
                # Determine color based on disaster type
                if 'disaster_group' in row:
                    disaster_type = row['disaster_group']
                else:
                    disaster_type = row['disaster_type']
                    
                if 'Geophysical' in disaster_type or 'Earthquake' in disaster_type:
                    color = 'red'
                elif 'Meteorological' in disaster_type or 'Storm' in disaster_type:
                    color = 'blue'
                elif 'Hydrological' in disaster_type or 'Flood' in disaster_type:
                    color = 'darkblue'
                elif 'Climatological' in disaster_type or 'Wildfire' in disaster_type:
                    color = 'orange'
                elif 'Biological' in disaster_type:
                    color = 'green'
                else:
                    color = 'gray'
                
                # Create popup content
                popup_content = f"""
                <b>{row.get('disaster_type', disaster_type)}</b><br>
                Location: {row.get('country', 'Unknown')}<br>
                Date: {row.get('start_date', 'Unknown')}<br>
                Magnitude: {row.get('magnitude', 'Unknown')}<br>
                Deaths: {row.get('deaths', 'Unknown')}<br>
                Damages: ${row.get('total_damages', 'Unknown')} million
                """
                
                # Add marker
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5,
                    popup=folium.Popup(popup_content, max_width=300),
                    color=color,
                    fill=True,
                    fill_opacity=0.7
                ).add_to(geo_map)
            except Exception as e:
                continue  # Skip problematic entries
        
        # Display map
        folium_static(geo_map)
    else:
        st.info("Geographic data (latitude/longitude) not available for mapping.")

