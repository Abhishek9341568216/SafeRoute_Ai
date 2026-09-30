import json
import plotly.express as px
import pandas as pd

def create_severity_distribution_chart(df):
    """Generates accident severity count chart."""
    if 'Severity' not in df.columns:
        return "{}"
        
    sev_counts = df['Severity'].value_counts().reset_index()
    sev_counts.columns = ['Severity', 'Count']
    sev_counts = sev_counts.sort_values('Severity')
    
    fig = px.bar(
        sev_counts, x='Severity', y='Count',
        title="Accident Severity Level Distribution (1=Minor to 4=Severe)",
        labels={'Severity': 'Accident Severity Rating', 'Count': 'Number of Recorded Incidents'},
        color='Count',
        color_continuous_scale='Reds'
    )
    fig.update_layout(template='plotly_white', margin=dict(l=40, r=40, t=50, b=40))
    return fig.to_json()

def create_hourly_accident_chart(df):
    """Generates hourly accident occurrence distribution."""
    if 'hour' not in df.columns and 'Start_Time' in df.columns:
        df['hour'] = pd.to_datetime(df['Start_Time']).dt.hour
        
    if 'hour' in df.columns:
        hourly_counts = df['hour'].value_counts().reset_index()
        hourly_counts.columns = ['Hour', 'Count']
        hourly_counts = hourly_counts.sort_values('Hour')
        
        fig = px.line(
            hourly_counts, x='Hour', y='Count',
            title="Accident Frequency by Time of Day (24-Hour Cycle)",
            labels={'Hour': 'Hour of Day (0-23)', 'Count': 'Total Incidents'},
            markers=True
        )
        fig.update_traces(line_color='#ef4444', line_width=3)
        fig.update_layout(template='plotly_white', margin=dict(l=40, r=40, t=50, b=40))
        return fig.to_json()
    return "{}"

def create_weather_accident_chart(df):
    """Generates accident count by weather condition."""
    if 'Weather_Condition' not in df.columns:
        return "{}"
        
    weather_counts = df['Weather_Condition'].value_counts().head(10).reset_index()
    weather_counts.columns = ['Weather_Condition', 'Count']
    
    fig = px.bar(
        weather_counts, x='Weather_Condition', y='Count',
        title="Top 10 Weather Conditions for Traffic Incidents",
        labels={'Weather_Condition': 'Weather Condition', 'Count': 'Incident Count'},
        color='Count',
        color_continuous_scale='Oranges'
    )
    fig.update_layout(template='plotly_white', margin=dict(l=40, r=40, t=50, b=40))
    return fig.to_json()
