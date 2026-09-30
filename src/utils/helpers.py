def classify_traffic_level(predicted_volume):
    """Categorizes traffic volume prediction into intuitive congestion levels."""
    if predicted_volume < 2000:
        return "Low Congestion (Free Flow)"
    elif predicted_volume <= 4200:
        return "Moderate Congestion"
    elif predicted_volume <= 5800:
        return "Heavy Traffic Congestion"
    else:
        return "Severe Gridlock Alert"

def classify_risk_level(accident_prob):
    """Categorizes accident risk probability into risk tiers."""
    if accident_prob < 0.25:
        return "Low Risk"
    elif accident_prob < 0.55:
        return "Moderate Risk"
    elif accident_prob < 0.80:
        return "High Risk"
    else:
        return "Critical Risk Alert"

def generate_decision_recommendations(traffic_volume, traffic_level, accident_prob, risk_level, weather, hour, is_weekend):
    """
    Generates tailored decision-support recommendations based on ML inference.
    """
    recs = []
    
    # Congestion recommendations
    if "Heavy" in traffic_level or "Gridlock" in traffic_level:
        recs.append("Traffic Management: Activate variable message signs (VMS) to suggest alternative arterial routes.")
        recs.append("Commuter Advice: Expect delays of 15-30 minutes. Consider shifting departure times by 45 minutes.")
    else:
        recs.append("Traffic Flow: Traffic conditions are within normal baseline capacity.")
        
    # Accident Risk & Environmental recommendations
    if risk_level in ["High Risk", "Critical Risk Alert"]:
        recs.append("Safety Dispatch: Increase highway patrol monitoring in high-density corridors.")
        recs.append("Driver Alert: Reduce speed limit by 10 mph and maintain extended safe braking distance.")
    elif risk_level == "Moderate Risk":
        recs.append("Safety Notice: Moderate accident risk detected. Exercise heightened awareness.")
        
    weather_lower = str(weather).lower()
    if any(w in weather_lower for w in ['rain', 'drizzle', 'snow', 'ice', 'fog', 'thunderstorm']):
        recs.append(f"Adverse Weather Notice ({weather}): Wet/slippery surface conditions detected. Turn on headlights and traction control.")
        
    if (7 <= hour <= 9) or (16 <= hour <= 18):
        recs.append("Peak Hours: Morning/evening commute rush hour in effect.")
        
    return recs

SYSTEM_DISCLAIMER = (
    "DISCLAIMER: SafeRoute AI is an intelligent decision-support system intended for "
    "academic and analytical decision support. It provides probabilistic risk assessments "
    "and congestion forecasts to support preventive decision-making. It does not replace "
    "official traffic management directives or direct vehicle navigation."
)
