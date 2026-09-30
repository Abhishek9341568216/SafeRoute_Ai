import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def create_hourly_traffic_chart(df):
    """Generates Plotly chart for hourly traffic volume trends."""
    if 'hour' not in df.columns:
        df['hour'] = pd.to_datetime(df['date_time']).dt.hour
        
    hourly_avg = df.groupby('hour')['traffic_volume'].mean().reset_index()
    
    fig = px.line(
        hourly_avg, x='hour', y='traffic_volume',
        title="Average Hourly Traffic Volume (Vehicles / Hour)",
        labels={'hour': 'Hour of Day (0-23)', 'traffic_volume': 'Average Traffic Volume'},
        markers=True
    )
    fig.update_traces(line_color='#2563eb', line_width=3)
    fig.update_layout(template='plotly_white', margin=dict(l=40, r=40, t=50, b=40))
    return fig.to_json()

def create_traffic_distribution_chart(df):
    """Generates histogram of traffic volume distribution."""
    fig = px.histogram(
        df, x='traffic_volume', nbins=30,
        title="Traffic Volume Distribution",
        labels={'traffic_volume': 'Traffic Volume (Vehicles/hr)'},
        color_discrete_sequence=['#3b82f6']
    )
    fig.update_layout(template='plotly_white', margin=dict(l=40, r=40, t=50, b=40))
    return fig.to_json()

def create_weather_traffic_chart(df):
    """Generates traffic volume by weather condition."""
    if 'weather_main' not in df.columns:
        return "{}"
        
    weather_avg = df.groupby('weather_main')['traffic_volume'].mean().reset_index().sort_values('traffic_volume', ascending=False)
    fig = px.bar(
        weather_avg, x='weather_main', y='traffic_volume',
        title="Traffic Volume Impact by Weather Condition",
        labels={'weather_main': 'Weather Condition', 'traffic_volume': 'Mean Traffic Volume'},
        color='traffic_volume',
        color_continuous_scale='Blues'
    )
    fig.update_layout(template='plotly_white', margin=dict(l=40, r=40, t=50, b=40))
    return fig.to_json()
