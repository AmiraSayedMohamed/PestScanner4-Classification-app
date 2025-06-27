
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import time
import pandas as pd
import requests
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import random
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
st.secrets["OPENROUTER_API_KEY"]

# Check if API key is loaded
if not OPENROUTER_API_KEY:
    st.error("OpenRouter API key not found. Please check your .env file or Streamlit secrets.")
    st.stop()

# App title and description
st.set_page_config(page_title="PestScanner Classification App", page_icon=":herb:")
st.title("🌱 PestScanner Disease Classification App")

# Sidebar info - Team Members
st.sidebar.header("Team Members")
st.sidebar.info(
    """
    **Abdelrahman**  
    Team Leader 
    
    **Amira**  
    Software Member
    
    **Omar**  
    Mechanical Member  
    """
)

# Green info message about what the app does - now in sidebar
st.sidebar.success("""
This app uses AI to analyze citrus leaf images and detect diseases.
Currently identifies:
- Black spot
- Citrus canker
""")

# Location selection in sidebar
st.sidebar.header("Location Information")
country = "Egypt"

# List of major Egyptian cities with coordinates
egyptian_cities = {
    "Cairo": (30.0444, 31.2357),
    "Alexandria": (31.2001, 29.9187),
    "Giza": (30.0131, 31.2089),
    "Shubra El-Kheima": (30.1286, 31.2422),
    "Port Said": (31.2565, 32.2841),
    "Suez": (29.9668, 32.5498),
    "Luxor": (25.6872, 32.6396),
    "Mansoura": (31.0409, 31.3785),
    "El-Mahalla El-Kubra": (30.9697, 31.1662),
    "Tanta": (30.7825, 31.0039),
    "Asyut": (27.1783, 31.1859),
    "Ismailia": (30.6043, 32.2723),
    "Faiyum": (29.3084, 30.8441),
    "Zagazig": (30.5877, 31.5020),
    "Aswan": (24.0889, 32.8998),
    "Damietta": (31.4167, 31.8144),
    "Damanhur": (31.0411, 30.4735),
    "Minya": (28.1099, 30.7503),
    "Beni Suef": (29.0667, 31.0833),
    "Qena": (26.1644, 32.7267),
    "Sohag": (26.5569, 31.6948),
    "Hurghada": (27.2579, 33.8116),
    "6th of October City": (29.9386, 30.9131),
    "Shibin El Kom": (30.5549, 31.0126),
    "Banha": (30.4667, 31.1833),
    "Arish": (31.1316, 33.7984),
    "10th of Ramadan City": (30.2994, 31.7417),
    "Kafr El Sheikh": (31.1117, 30.9394),
    "Marsa Matruh": (31.3525, 27.2453),
    "Idfu": (24.9781, 32.8789),
    "Mit Ghamr": (30.7167, 31.2500),
    "Al-Hamidiyya": (31.1167, 30.9333),
    "Desouk": (31.1325, 30.6478),
    "Qalyub": (30.1800, 31.2064),
    "Abu Kabir": (30.7250, 31.6714),
    "Kafr El Dawwar": (31.1339, 30.1297),
    "Girga": (26.3333, 31.9000),
    "Akhmim": (26.5667, 31.7500),
    "Matareya": (30.1833, 31.4667)
}

# City selection dropdown
selected_city = st.sidebar.selectbox("Select your city in Egypt:", list(egyptian_cities.keys()))

# Display selected location in sidebar
st.sidebar.success(f"📍 Selected Location: {selected_city}, {country}")

# Combine into location string for weather API
location = f"{selected_city}, {country}"

# Weather API Configuration
OPENWEATHER_API_KEY = "77e0c3ce19e37b2f9c0a39cea77a7d19"

# Load your trained model (silently)
@st.cache_resource
def load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path='plant_disease_classifier_quant.tflite')
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

model = load_model()

# Define your class names
class_names = ['black-spot', 'citrus-canker']

# Recommendation database based on your uploaded images
recommendation_db = {
    'black-spot': {
        'chemical': [
            {'name': 'Ortus super', 'active': 'Fenpyroximate 5% EC', 'type': 'Contact', 'safety': 'Wear gloves and mask. Avoid application in windy conditions.'},
            {'name': 'TAK', 'active': 'Chlorpyrifos 48% EC', 'type': 'Systemic', 'safety': 'Highly toxic to bees. Apply in evening when bees are less active.'}
        ],
        'organic': [
            'Neem oil spray (apply every 7-10 days)',
            'Baking soda solution (1 tbsp baking soda + 1 tsp vegetable oil + 1 gallon water)',
            'Copper-based fungicides'
        ],
        'cultural': [
            'Remove and destroy infected leaves',
            'Improve air circulation by pruning',
            'Avoid overhead watering',
            'Rotate with non-citrus crops for 2 seasons'
        ],
        'description': 'Black spot is a fungal disease that causes dark spots on leaves and fruit. It thrives in warm, wet conditions.',
        'weather_risk': {
            'high_humidity': 70,
            'optimal_temp_range': (20, 30),
            'rain_risk': True
        }
    },
    'citrus-canker': {
        'chemical': [
            {'name': 'Biomectin', 'active': 'Abamectin 3% EC', 'type': 'Systemic', 'safety': 'Use protective clothing. Do not apply near water sources.'},
            {'name': 'AVENUE', 'active': 'Imidacloprid 70% SC', 'type': 'Systemic', 'safety': 'Toxic to aquatic organisms. Keep away from waterways.'}
        ],
        'organic': [
            'Copper-based bactericides',
            'Streptomycin sulfate (antibiotic spray)',
            'Garlic and chili pepper extract sprays'
        ],
        'cultural': [
            'Remove and burn infected plants',
            'Disinfect tools with 10% bleach solution',
            'Plant resistant varieties when available',
            'Implement strict quarantine measures for new plants'
        ],
        'description': 'Citrus canker is a bacterial disease causing raised lesions on leaves, stems, and fruit. Highly contagious.',
        'weather_risk': {
            'high_humidity': 75,
            'optimal_temp_range': (25, 35),
            'rain_risk': True
        }
    }
}

def generate_mock_pest_reports(city, num_reports=15):
    """Generate mock pest reports around a city with fallback coordinates"""
    try:
        if city in egyptian_cities:
            base_lat, base_lon = egyptian_cities[city]
        else:
            geolocator = Nominatim(user_agent="pest_monitoring_app", timeout=10)
            location = geolocator.geocode(city + ", Egypt")
            if not location:
                return []
            base_lat, base_lon = location.latitude, location.longitude
        
        reports = []
        diseases = ['black-spot', 'citrus-canker']
        
        for _ in range(num_reports):
            lat = base_lat + random.uniform(-0.2, 0.2)
            lon = base_lon + random.uniform(-0.2, 0.2)
            report_date = datetime.now() - timedelta(days=random.randint(0, 30))
            disease = random.choice(diseases)
            severity = random.choice(['low', 'medium', 'high'])
            
            reports.append({
                'latitude': lat,
                'longitude': lon,
                'disease': disease,
                'severity': severity,
                'date': report_date.strftime('%Y-%m-%d'),
                'reporter': f"Farmer {random.randint(1, 100)}"
            })
        
        return reports
    except Exception as e:
        st.error(f"Error generating mock reports: {str(e)}")
        return []

def create_pest_map(city, reports):
    """Create an interactive pest monitoring map with fallback coordinates"""
    try:
        if city in egyptian_cities:
            lat, lon = egyptian_cities[city]
        else:
            geolocator = Nominatim(user_agent="pest_monitoring_app", timeout=10)
            location = geolocator.geocode(city + ", Egypt")
            if not location:
                return None
            lat, lon = location.latitude, location.longitude
        
        m = folium.Map(location=[lat, lon], zoom_start=10)
        
        folium.Marker(
            [lat, lon],
            popup=f"<b>{city}</b>",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)
        
        if reports:
            for report in reports:
                color = 'red' if report['disease'] == 'citrus-canker' else 'orange'
                icon = folium.Icon(color=color, icon='bug', prefix='fa')
                popup_content = f"""
                <div style="width: 200px;">
                    <h4 style="margin-bottom: 5px;">{report['disease'].replace('-', ' ').title()}</h4>
                    <p><b>Severity:</b> {report['severity'].title()}</p>
                    <p><b>Reported:</b> {report['date']}</p>
                    <p><b>By:</b> {report['reporter']}</p>
                </div>
                """
                folium.Marker(
                    [report['latitude'], report['longitude']],
                    popup=popup_content,
                    icon=icon
                ).add_to(m)
            
            from folium.plugins import HeatMap
            heat_data = [[report['latitude'], report['longitude'], 
                         1 if report['severity'] == 'low' else 
                         2 if report['severity'] == 'medium' else 3] 
                        for report in reports]
            HeatMap(heat_data, radius=20, blur=15).add_to(m)
        
        return m
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        return None

def get_weather_forecast(location):
    """Get 5-day weather forecast from OpenWeatherMap API"""
    try:
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={OPENWEATHER_API_KEY}"
        geo_response = requests.get(geo_url).json()
        
        if not geo_response:
            st.error("Location not found. Please try again.")
            return None
            
        lat = geo_response[0]['lat']
        lon = geo_response[0]['lon']
        
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(forecast_url).json()
        
        if response.get('cod') != '200':
            st.error("Error fetching weather data. Please try again later.")
            return None
            
        forecast_data = []
        for item in response['list']:
            date = datetime.fromtimestamp(item['dt'])
            forecast_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'time': date.strftime('%H:%M'),
                'temp': item['main']['temp'],
                'humidity': item['main']['humidity'],
                'weather': item['weather'][0]['main'],
                'rain': item.get('rain', {}).get('3h', 0)
            })
        
        return forecast_data
    except Exception as e:
        st.error(f"Error fetching weather data: {str(e)}")
        return None

def analyze_weather_risk(disease, forecast_data):
    """Analyze weather conditions for disease risk with more detailed output"""
    if not forecast_data or disease not in recommendation_db:
        return None
    
    weather_risk = recommendation_db[disease]['weather_risk']
    risk_factors = {
        'high_humidity': {'count': 0, 'max': 0, 'days': set()},
        'optimal_temp': {'count': 0, 'days': set(), 'min': float('inf'), 'max': -float('inf')},
        'rain': {'count': 0, 'days': set()}
    }
    
    for day in forecast_data:
        date = day['date']
        if day['humidity'] > weather_risk['high_humidity']:
            risk_factors['high_humidity']['count'] += 1
            risk_factors['high_humidity']['days'].add(date)
            risk_factors['high_humidity']['max'] = max(risk_factors['high_humidity']['max'], day['humidity'])
        temp_range = weather_risk['optimal_temp_range']
        if temp_range[0] <= day['temp'] <= temp_range[1]:
            risk_factors['optimal_temp']['count'] += 1
            risk_factors['optimal_temp']['days'].add(date)
            risk_factors['optimal_temp']['min'] = min(risk_factors['optimal_temp']['min'], day['temp'])
            risk_factors['optimal_temp']['max'] = max(risk_factors['optimal_temp']['max'], day['temp'])
        if weather_risk['rain_risk'] and day.get('rain', 0) > 0:
            risk_factors['rain']['count'] += 1
            risk_factors['rain']['days'].add(date)
    
    messages = []
    temp_range = weather_risk['optimal_temp_range']
    if risk_factors['optimal_temp']['count'] > 0:
        day_count = len(risk_factors['optimal_temp']['days'])
        messages.append(
            f"🌡️ Ideal temperatures ({temp_range[0]}°C-{temp_range[1]}°C) expected on {day_count} days "
            f"(actual: {risk_factors['optimal_temp']['min']}°C to {risk_factors['optimal_temp']['max']}°C)"
        )
    if risk_factors['high_humidity']['count'] > 0:
        day_count = len(risk_factors['high_humidity']['days'])
        messages.append(
            f"💧 High humidity (> {weather_risk['high_humidity']}%) expected on {day_count} days "
            f"(peaking at {risk_factors['high_humidity']['max']}%)"
        )
    if risk_factors['rain']['count'] > 0:
        day_count = len(risk_factors['rain']['days'])
        messages.append(f"☔ Rain expected on {day_count} days - will favor fungal growth")
    
    return messages if messages else None

def display_recommendations(disease, location=None):
    """Display recommendations based on the detected disease"""
    if disease not in recommendation_db:
        st.warning("No recommendations available for this disease.")
        return
    
    data = recommendation_db[disease]
    
    st.subheader(f"🌱 {disease.replace('-', ' ').title()} Information")
    st.info(data['description'])
    
    st.markdown("---")
    
    st.subheader("🧪 Chemical Control Options")
    if data['chemical']:
        chem_df = pd.DataFrame(data['chemical'])
        st.table(chem_df)
        st.warning("⚠️ Always follow pesticide label instructions and local regulations")
    else:
        st.info("No chemical recommendations available")
    
    st.subheader("🍃 Organic/Natural Remedies")
    for remedy in data['organic']:
        st.markdown(f"- {remedy}")
    
    st.subheader("🌿 Cultural Practices")
    for practice in data['cultural']:
        st.markdown(f"- {practice}")
    
    st.markdown("---")
    st.subheader("📜 Regulatory Information")
    st.info("""
    - Always check with your local agricultural extension office
    - Some pesticides may be restricted in your area
    - Follow recommended pre-harvest intervals
    """)

def preprocess_image(image):
    """Preprocess the uploaded image for model inference"""
    img = np.array(image)
    if img.shape[-1] == 4:  # RGBA case
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif len(img.shape) == 2:  # Grayscale case
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)

def get_openrouter_response(question, max_retries=3):
    """Fetch response from OpenRouter API with enhanced error handling"""
    import time
    for attempt in range(max_retries):
        try:
            headers = {
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://your-site-url.com",  # Replace or remove
                "X-Title": "PestScanner App",  # Replace or remove
            }
            data = {
                "model": "deepseek/deepseek-r1:free",
                "messages": [
                    {"role": "system", "content": "You are a helpful agricultural expert specializing in pest and disease management. Provide concise and practical advice based on the context of citrus farming in Egypt."},
                    {"role": "user", "content": question}
                ],
                "max_tokens": 1000,
                "temperature": 0.7,
                "usage": {"include": True}
            }
            print(f"Attempt {attempt + 1}: Sending request to OpenRouter...")
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            print(f"Response status code: {response.status_code}")
            response.raise_for_status()
            result = response.json()
            print(f"Response received: {result}")
            content = result["choices"][0]["message"]["content"]
            if not content and "reasoning" in result["choices"][0]["message"]:
                content = "Response truncated. Increase max_tokens or try again. Reasoning: " + result["choices"][0]["message"]["reasoning"]
            # Limit response to 2000 characters to prevent overflow
            if len(content) > 2000:
                content = content[:2000] + " [Response truncated due to character limit]"
            return content
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {str(e)} - Response: {response.text if 'response' in locals() else 'No response'}")
            if response.status_code == 400 and "not a valid model ID" in str(e):
                return f"Error: Invalid model ID 'deepseek/deepseek-r1:free'. Please check available models on OpenRouter."
            elif response.status_code == 429:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            return f"Error: {str(e)} - Response: {response.text if 'response' in locals() else 'No response'}"
        except requests.exceptions.RequestException as e:
            print(f"Request Exception: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return f"Error: {str(e)} - Response: {getattr(e.response, 'text', 'No response')}"
        except Exception as e:
            print(f"Unexpected Error: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return f"Error: {str(e)} - No response received"
    return "Error: Max retries reached. Please try again later."

# Chatbot section
st.sidebar.header("Ask an Expert")
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
user_question = st.sidebar.text_input("Enter your question (e.g., 'How do I treat powdery mildew organically?')")
if user_question and user_question != st.session_state.last_question:
    st.session_state.last_question = user_question
    with st.sidebar:
        st.write("**Response:**")
        with st.spinner("Fetching expert advice..."):
            response = get_openrouter_response(user_question)
        st.write(response)

# File uploader
uploaded_file = st.file_uploader("Choose a citrus leaf image...", type=["jpg", "jpeg", "png"])

# Add guidance when no file is uploaded
if model is not None and uploaded_file is None:
    st.info("ℹ️ Please upload a citrus leaf image to get a diagnosis")

if model is not None and uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with st.spinner("Analyzing the leaf..."):
            processed_image = preprocess_image(image)
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            if processed_image.shape != tuple(input_details[0]['shape']):
                processed_image = np.resize(processed_image, input_details[0]['shape'])
            model.set_tensor(input_details[0]['index'], processed_image)
            model.invoke()
            predictions = model.get_tensor(output_details[0]['index'])
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            time.sleep(1)
        
        st.success("Analysis Complete!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Diagnosis")
            disease = class_names[predicted_class]
            st.write(f"**Detected Disease:** {disease}")
            st.write(f"**Confidence:** {confidence:.2%}")
            st.warning("⚠️ Disease detected!")
            with st.expander("🛠️ View Treatment Recommendations", expanded=True):
                display_recommendations(disease, location)
        
        with col2:
            st.subheader("Probability Distribution")
            chart_data = pd.DataFrame({"Disease": class_names, "Probability": predictions[0]})
            st.bar_chart(chart_data.set_index("Disease"))
            
            st.subheader("🌤️ Disease Risk Forecasting")
            st.write("""
            This section analyzes local weather conditions to assess disease spread risk.
            The forecast checks for:
            - Ideal temperature ranges for disease development
            - High humidity levels that favor infection
            - Rain events that can spread pathogens
            """)
            if location:
                with st.spinner("Fetching weather forecast..."):
                    forecast_data = get_weather_forecast(location)
                    if forecast_data:
                        risk_factors = analyze_weather_risk(disease, forecast_data)
                        if risk_factors:
                            st.warning("⚠️ High Risk Conditions Detected")
                            for factor in risk_factors:
                                st.write(f"- {factor}")
                            st.info("""
                            **Recommended Actions:**
                            - Increase monitoring of plants
                            - Apply preventive treatments
                            - Improve air circulation
                            - Remove infected plant material
                            """)
                        else:
                            st.success("✅ Current weather shows low disease risk")
                    else:
                        st.warning("Weather data unavailable - using general recommendations")
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")

# Pest Monitoring Map Section
st.markdown("---")
st.header("🌍 Pest Monitoring Map")
st.write("Track nearby pest reports and outbreaks.")

@st.cache_data
def get_cached_reports(city):
    return generate_mock_pest_reports(city)

use_gps = st.checkbox("Use GPS (if available)", key="gps_toggle")

reports = get_cached_reports(selected_city)

map_container = st.container()
with map_container:
    if reports:
        pest_map = create_pest_map(selected_city, reports)
        if pest_map:
            st_folium(pest_map, width=700, height=500, key="pest_map")

if reports:
    df_reports = pd.DataFrame(reports)
    st.subheader("📊 Report Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(df_reports['disease'].value_counts())
    with col2:
        st.bar_chart(df_reports['severity'].value_counts())
    st.dataframe(df_reports.sort_values('date', ascending=False).head(5), hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <style>
    .footer {
        font-size: 12px;
        color: #777;
        text-align: center;
    }
    </style>
    <div class="footer">
        Citrus Disease Classifier | Made with Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
