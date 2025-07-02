import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style and figure parameters to prevent overlap
plt.style.use('ggplot')
sns.set_palette('viridis')
plt.rcParams['figure.autolayout'] = True  # Automatically adjust subplots
plt.rcParams['figure.constrained_layout.use'] = True  # Use constrained layout

## 1. Data Loading and Preparation
def load_data():
    """Load the dataset or create sample data if not available"""
    try:
        df = pd.read_csv('US_Accidents_Dataset.csv')
        print("Real dataset loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df['Start_Time'].min()} to {df['Start_Time'].max()}")
        return df
    except FileNotFoundError:
        print("Dataset not found. Creating sample data for demonstration...")
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=5000, freq='H')
        cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
                 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
        states = ['CA', 'TX', 'NY', 'FL', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']

        sample_data = {
            'ID': range(5000),
            'Severity': np.random.choice([1, 2, 3, 4], size=5000, p=[0.1, 0.6, 0.2, 0.1]),
            'Start_Time': dates,
            'End_Time': dates + pd.to_timedelta(np.random.randint(1, 120, size=5000), unit='m'),
            'Weather_Condition': np.random.choice(['Clear', 'Rain', 'Snow', 'Fog', 'Cloudy', 'Thunderstorm'],
                                               size=5000, p=[0.5, 0.2, 0.1, 0.05, 0.1, 0.05]),
            'Sunrise_Sunset': np.random.choice(['Day', 'Night'], size=5000, p=[0.7, 0.3]),
            'City': np.random.choice(cities, size=5000),
            'State': np.random.choice(states, size=5000),
            'Start_Lat': np.random.uniform(25, 49, size=5000),
            'Start_Lng': np.random.uniform(-125, -67, size=5000),
            'Temperature(F)': np.random.uniform(10, 100, size=5000),
            'Visibility(mi)': np.random.uniform(0.1, 10, size=5000),
            'Wind_Speed(mph)': np.random.uniform(0, 30, size=5000),
            'Precipitation(in)': np.random.exponential(0.1, size=5000)
        }
        df = pd.DataFrame(sample_data)
        print("Sample data created with realistic distributions")
        return df

df = load_data()

# Data cleaning and feature engineering
def prepare_data(df):
    """Prepare the data for analysis"""
    df['Start_Time'] = pd.to_datetime(df['Start_Time'])
    df['Hour'] = df['Start_Time'].dt.hour
    df['Weekday'] = df['Start_Time'].dt.day_name()
    df['Month'] = df['Start_Time'].dt.month_name()
    df['Year'] = df['Start_Time'].dt.year
    df['Weather_Condition'] = df['Weather_Condition'].fillna('Unknown')
    df['Sunrise_Sunset'] = df['Sunrise_Sunset'].fillna('Unknown')
    df['Severity_Level'] = pd.cut(df['Severity'], bins=[0, 1, 2, 3, 4, 5],
                                labels=['Minor', 'Moderate', 'Serious', 'Severe', 'Fatal'],
                                right=False)
    return df

df = prepare_data(df)

## 2. Temporal Analysis with Proper Spacing
def plot_temporal_analysis(df):
    """Visualize temporal patterns with proper spacing"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Hourly distribution
    sns.countplot(ax=axes[0, 0], x='Hour', data=df)
    axes[0, 0].set_title('Accidents by Hour of Day', fontsize=14)
    axes[0, 0].set_xlabel('Hour of Day', fontsize=12)
    axes[0, 0].set_ylabel('Number of Accidents', fontsize=12)

    # Weekly distribution
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sns.countplot(ax=axes[0, 1], x='Weekday', data=df, order=weekday_order)
    axes[0, 1].set_title('Accidents by Day of Week', fontsize=14)
    axes[0, 1].set_xlabel('Day of Week', fontsize=12)
    axes[0, 1].set_ylabel('Number of Accidents', fontsize=12)
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Monthly distribution
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                 'July', 'August', 'September', 'October', 'November', 'December']
    sns.countplot(ax=axes[1, 0], x='Month', data=df, order=month_order)
    axes[1, 0].set_title('Accidents by Month', fontsize=14)
    axes[1, 0].set_xlabel('Month', fontsize=12)
    axes[1, 0].set_ylabel('Number of Accidents', fontsize=12)
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Severity by hour
    sns.boxplot(ax=axes[1, 1], x='Severity', y='Hour', data=df)
    axes[1, 1].set_title('Accident Severity by Hour of Day', fontsize=14)
    axes[1, 1].set_xlabel('Severity Level', fontsize=12)
    axes[1, 1].set_ylabel('Hour of Day', fontsize=12)

    plt.tight_layout(pad=3.0)  # Add extra padding between subplots
    plt.show()

plot_temporal_analysis(df)

## 3. Weather Analysis with Proper Spacing
def plot_weather_analysis(df):
    """Analyze weather factors with proper spacing"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Weather conditions
    weather_counts = df['Weather_Condition'].value_counts().nlargest(10)
    sns.barplot(ax=axes[0, 0], x=weather_counts.values, y=weather_counts.index)
    axes[0, 0].set_title('Top 10 Weather Conditions', fontsize=14)
    axes[0, 0].set_xlabel('Number of Accidents', fontsize=12)
    axes[0, 0].set_ylabel('Weather Condition', fontsize=12)

    # Severity by weather
    weather_severity = df.groupby('Weather_Condition')['Severity'].mean().nlargest(10)
    sns.barplot(ax=axes[0, 1], x=weather_severity.values, y=weather_severity.index)
    axes[0, 1].set_title('Weather Conditions by Average Severity', fontsize=14)
    axes[0, 1].set_xlabel('Average Severity', fontsize=12)
    axes[0, 1].set_ylabel('Weather Condition', fontsize=12)

    # Daylight conditions
    sns.countplot(ax=axes[1, 0], x='Sunrise_Sunset', hue='Severity_Level', data=df)
    axes[1, 0].set_title('Accidents by Daylight Condition', fontsize=14)
    axes[1, 0].set_xlabel('Daylight Condition', fontsize=12)
    axes[1, 0].set_ylabel('Number of Accidents', fontsize=12)
    axes[1, 0].legend(title='Severity Level', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Temperature distribution
    if 'Temperature(F)' in df.columns:
        sns.histplot(ax=axes[1, 1], x=df['Temperature(F)'], bins=30, kde=True)
        axes[1, 1].set_title('Temperature Distribution', fontsize=14)
        axes[1, 1].set_xlabel('Temperature (F)', fontsize=12)
        axes[1, 1].set_ylabel('Number of Accidents', fontsize=12)

    plt.tight_layout(pad=3.0)  # Add extra padding
    plt.show()

plot_weather_analysis(df)

## 4. Geospatial Analysis (unchanged)
def create_accident_map(df):
    """Create interactive map of accident hotspots"""
    print("\n Creating accident hotspot map...")
    sample_df = df.sample(n=2000, random_state=42) if len(df) > 2000 else df
    avg_lat = sample_df['Start_Lat'].mean()
    avg_lng = sample_df['Start_Lng'].mean()
    m = folium.Map(location=[avg_lat, avg_lng], zoom_start=5, tiles='CartoDB dark_matter')
    heat_data = [[row['Start_Lat'], row['Start_Lng']] for _, row in sample_df.iterrows()]
    HeatMap(heat_data, radius=12, blur=15, max_zoom=13).add_to(m)

    top_cities = df['City'].value_counts().nlargest(5).index
    for city in top_cities:
        city_df = df[df['City'] == city].sample(min(50, len(df[df['City'] == city])))
        for _, row in city_df.iterrows():
            popup_text = f"""
            <b>City:</b> {row['City']}, {row['State']}<br>
            <b>Severity:</b> {row['Severity']}<br>
            <b>Weather:</b> {row['Weather_Condition']}<br>
            <b>Time:</b> {row['Start_Time']}
            """
            folium.CircleMarker(
                location=[row['Start_Lat'], row['Start_Lng']],
                radius=row['Severity'],
                color='red' if row['Severity'] > 3 else 'orange',
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(popup_text, max_width=250)
            ).add_to(m)

    m.save('accident_hotspots.html')
    print("Hotspot map saved as 'accident_hotspots.html'")

create_accident_map(df)

## 5. Cross-Factor Analysis with Proper Spacing
def plot_cross_factor_analysis(df):
    """Analyze relationships between factors with proper spacing"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Weather vs time of day
    weather_hour = df.groupby(['Hour', 'Weather_Condition']).size().unstack().fillna(0)
    top_weather = df['Weather_Condition'].value_counts().nlargest(3).index
    weather_hour[top_weather].plot(ax=axes[0, 0])
    axes[0, 0].set_title('Accidents by Hour for Top 3 Weather Conditions', fontsize=14)
    axes[0, 0].set_xlabel('Hour of Day', fontsize=12)
    axes[0, 0].set_ylabel('Number of Accidents', fontsize=12)
    axes[0, 0].legend(title='Weather Condition')

    # Severity by weather and temperature
    if 'Temperature(F)' in df.columns:
        sns.scatterplot(ax=axes[0, 1], x='Temperature(F)', y='Severity',
                       hue='Weather_Condition', data=df, alpha=0.6)
        axes[0, 1].set_title('Severity by Temperature and Weather', fontsize=14)
        axes[0, 1].set_xlabel('Temperature (F)', fontsize=12)
        axes[0, 1].set_ylabel('Severity Level', fontsize=12)
        axes[0, 1].legend(title='Weather Condition', bbox_to_anchor=(1.05, 1), loc='upper left')

    # City analysis
    if 'City' in df.columns:
        city_counts = df['City'].value_counts().nlargest(10)
        sns.barplot(ax=axes[1, 0], x=city_counts.values, y=city_counts.index)
        axes[1, 0].set_title('Top 10 Cities by Accident Count', fontsize=14)
        axes[1, 0].set_xlabel('Number of Accidents', fontsize=12)
        axes[1, 0].set_ylabel('City', fontsize=12)

    # State analysis
    if 'State' in df.columns:
        state_severity = df.groupby('State')['Severity'].mean().sort_values(ascending=False)
        sns.barplot(ax=axes[1, 1], x=state_severity.values, y=state_severity.index)
        axes[1, 1].set_title('Average Severity by State', fontsize=14)
        axes[1, 1].set_xlabel('Average Severity', fontsize=12)
        axes[1, 1].set_ylabel('State', fontsize=12)

    plt.tight_layout(pad=3.0)  # Add extra padding
    plt.show()

plot_cross_factor_analysis(df)

## 6. Summary Statistics (unchanged)
def show_summary_stats(df):
    print("\n SUMMARY STATISTICS")
    print("="*40)
    print("\n TEMPORAL DISTRIBUTION")
    print("-"*30)
    print(f"Most common hour: {df['Hour'].mode()[0]}:00")
    print(f"Most common weekday: {df['Weekday'].mode()[0]}")
    print(f"Most common month: {df['Month'].mode()[0]}")

    print("\n WEATHER CONDITIONS")
    print("-"*30)
    print(f"Most common weather: {df['Weather_Condition'].mode()[0]}")
    print(f"Weather with highest severity: {df.groupby('Weather_Condition')['Severity'].mean().idxmax()}")

    if 'City' in df.columns:
        print("\n LOCATION ANALYSIS")
        print("-"*30)
        print(f"City with most accidents: {df['City'].value_counts().idxmax()}")
        print(f"State with most accidents: {df['State'].value_counts().idxmax()}")
        print(f"State with highest severity: {df.groupby('State')['Severity'].mean().idxmax()}")

    print("\n SEVERITY ANALYSIS")
    print("-"*30)
    print(f"Average severity: {df['Severity'].mean():.2f}")
    print(f"Severity distribution:\n{df['Severity_Level'].value_counts(normalize=True).mul(100).round(1)}")

show_summary_stats(df)
