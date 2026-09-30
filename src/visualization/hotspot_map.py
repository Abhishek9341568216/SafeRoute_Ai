import folium
from folium.plugins import MarkerCluster, HeatMap

def generate_leaflet_map_html(hotspots):
    """
    Generates interactive Leaflet Map HTML centered at the mean coordinates of detected clusters.
    """
    if not hotspots:
        center_lat, center_lng = 44.9778, -93.2650  # Default Minneapolis center
    else:
        center_lat = sum(h['latitude'] for h in hotspots) / len(hotspots)
        center_lng = sum(h['longitude'] for h in hotspots) / len(hotspots)
        
    m = folium.Map(location=[center_lat, center_lng], zoom_start=6, tiles='OpenStreetMap')
    
    # Heatmap layer
    heat_data = [[h['latitude'], h['longitude'], h['accident_count']] for h in hotspots]
    if heat_data:
        HeatMap(heat_data, radius=25, blur=15, max_zoom=10).add_to(m)
        
    marker_cluster = MarkerCluster().add_to(m)
    
    for h in hotspots:
        color = 'red' if h['risk_level'] == 'High' else ('orange' if h['risk_level'] == 'Medium' else 'green')
        popup_html = f"""
        <div style="font-family: Arial; width: 180px;">
            <h4 style="margin-bottom: 5px; color: #1e293b;">Hotspot Cluster #{h['cluster_id']}</h4>
            <hr style="margin: 5px 0;">
            <b>Accident Count:</b> {h['accident_count']}<br>
            <b>Avg Severity:</b> {h.get('avg_severity', 'N/A')}<br>
            <b>Risk Level:</b> <span style="color: {color}; font-weight: bold;">{h['risk_level']}</span><br>
            <small>Lat: {h['latitude']}, Lng: {h['longitude']}</small>
        </div>
        """
        folium.Marker(
            location=[h['latitude'], h['longitude']],
            popup=folium.Popup(popup_html, max_width=220),
            icon=folium.Icon(color=color, icon='warning-sign')
        ).add_to(marker_cluster)
        
    return m._repr_html_()
