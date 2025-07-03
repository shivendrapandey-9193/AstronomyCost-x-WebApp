import subprocess
import sys
import streamlit as st
import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import nltk
from groq import Groq
import folium
from streamlit_folium import folium_static
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objs as go
from geopy.geocoders import Nominatim
import google.generativeai as genai
import logging
import time
import re
import os
from dotenv import load_dotenv

# --- Configure Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# --- Set Streamlit Page Config ---
st.set_page_config(page_title="AstroCostX", page_icon="🛰️", layout="wide")

# --- Auto-install Dependencies ---
def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package}: {e}")

try:
    import google.generativeai
except ImportError:
    install_package("google-generativeai")
    import google.generativeai as genai

try:
    from textblob import TextBlob
except ImportError:
    install_package("textblob")
    from textblob import TextBlob

try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    logger.warning(f"NLTK download failed: {e}")

# --- Load Environment Variables ---
# First try Streamlit Cloud secrets, then fall back to .env file
try:
    NASA_API_KEY = st.secrets["NASA_API_KEY"]
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    WEATHER_API_KEY = st.secrets["WEATHER_API_KEY"]
except KeyError as e:
    logger.warning(f"Streamlit secrets not found: {e}. Falling back to .env file.")
    load_dotenv()
    NASA_API_KEY = os.getenv("NASA_API_KEY")
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# --- API Key Dictionary ---
API_KEYS = {
    "NASA": NASA_API_KEY,
    "NEWS": NEWS_API_KEY,
    "GROQ": GROQ_API_KEY,
    "GOOGLE": GOOGLE_API_KEY,
    "WEATHER": WEATHER_API_KEY,
}

# --- API Key Validation ---
def is_valid_api_key(api_key, provider):
    if not api_key or api_key == f"your_{provider.lower()}_api_key":
        logger.error(f"Invalid or missing {provider} API key.")
        return False
    if provider == "GROQ":
        return bool(re.match(r'^gsk_[a-zA-Z0-9]+$', api_key))
    return True

# --- AI Client Initialization ---
def initialize_ai_clients():
    clients = {}

    # GROQ Client Initialization
    if is_valid_api_key(API_KEYS["GROQ"], "GROQ"):
        try:
            # 🚫 DO NOT PASS 'proxies'
            clients["groq"] = Groq(api_key=API_KEYS["GROQ"])
            logger.info("GROQ client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize GROQ client: {e}")
            clients["groq"] = None
    else:
        clients["groq"] = None
        logger.warning("GROQ API key invalid or missing.")

    # Gemini Client Initialization
    if is_valid_api_key(API_KEYS["GOOGLE"], "GOOGLE"):
        try:
            genai.configure(api_key=API_KEYS["GOOGLE"])
            clients["gemini"] = genai.GenerativeModel("gemini-1.5-flash")
            logger.info("Google Gemini client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Google Gemini client: {e}")
            clients["gemini"] = None
    else:
        clients["gemini"] = None
        logger.warning("Google Gemini API key invalid or missing.")

    return clients


# --- Initialize on Run ---
ai_clients = initialize_ai_clients()

# --- Token Usage Tracking ---
if "token_usage" not in st.session_state:
    st.session_state.token_usage = {
        "total_tokens": 0,
        "requests": 0,
        "last_reset": time.time(),
        "token_limit": 6000,
    }

def update_token_usage(response, provider):
    try:
        if provider == "groq" and hasattr(response, "usage") and response.usage:
            tokens = response.usage.total_tokens
            st.session_state.token_usage["total_tokens"] += tokens
            st.session_state.token_usage["requests"] += 1
            logger.info(f"Used {tokens} tokens in {provider} request. Total: {st.session_state.token_usage['total_tokens']}")
            if time.time() - st.session_state.token_usage["last_reset"] > 60:
                st.session_state.token_usage["total_tokens"] = 0
                st.session_state.token_usage["requests"] = 0
                st.session_state.token_usage["last_reset"] = time.time()
    except Exception as e:
        logger.error(f"Error tracking token usage for {provider}: {e}")

# --- AI Helper Functions ---
def groq_chat(messages, model="meta-llama-3-8b-instruct", max_retries=3):
    if not ai_clients.get("groq"):
        logger.warning("GROQ client unavailable, attempting Gemini fallback.")
        return gemini_fallback(messages)

    for attempt in range(max_retries):
        try:
            response = ai_clients["groq"].chat.completions.create(
                messages=messages,
                model=model,
                temperature=0.4,
                max_tokens=512,
            )
            if response and response.choices and response.choices[0].message.content:
                update_token_usage(response, "groq")
                return response.choices[0].message.content.strip(), "groq"
            else:
                logger.warning(f"GROQ returned empty or invalid response on attempt {attempt + 1}")
        except Exception as e:
            logger.error(f"GROQ API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                break
            time.sleep(2 ** attempt)
    logger.warning("GROQ retries exhausted, attempting Gemini fallback.")
    return gemini_fallback(messages)

def gemini_fallback(messages):
    if not ai_clients.get("gemini"):
        logger.error("Google Gemini client unavailable.")
        return "AI service unavailable due to configuration issues.", "none"
    
    try:
        response = ai_clients["gemini"].generate_content(
            messages[-1]["content"],
            generation_config={"max_output_tokens": 512, "temperature": 0.4}
        )
        if response and response.text:
            logger.info("Successfully used Google Gemini as fallback.")
            return response.text.strip(), "gemini"
        else:
            logger.warning("Google Gemini returned empty or invalid response.")
            return "AI service unavailable due to empty response.", "none"
    except Exception as e:
        logger.error(f"Google Gemini API error: {e}")
        return "AI service unavailable due to API issues.", "none"

# --- Theme Configuration ---
THEMES = {
    "Dark": {
        "background": "linear-gradient(135deg, #0B0B3B 0%, #1C2526 50%, #2E3A59 100%), url('https://www.transparenttextures.com/patterns/stardust.png')",
        "text_color": "#E6E6FA",
        "primary_color": "#00B7EB",
        "secondary_color": "#FF4D4D",
        "button_bg": "#00B7EB",
        "button_hover": "#FF4D4D",
        "card_bg": "rgba(20, 20, 40, 0.8)",
        "header_gradient": "linear-gradient(to right, #00B7EB, #FF4D4D)",
        "weather_gradient": "linear-gradient(90deg, #00B7EB33, #FF4D4D22, #40C4FF33)",
        "table_bg": "linear-gradient(90deg, #2E3A59, #1C2526)",
        "contrast_text": "#FFFFFF",
    },
    "Light": {
        "background": "linear-gradient(135deg, #F0F8FF 0%, #ADD8E6 100%)",
        "text_color": "#1C2526",
        "primary_color": "#4682B4",
        "secondary_color": "#DC143C",
        "button_bg": "#4682B4",
        "button_hover": "#DC143C",
        "card_bg": "rgba(255, 255, 255, 0.95)",
        "header_gradient": "linear-gradient(to right, #4682B4, #DC143C)",
        "weather_gradient": "linear-gradient(90deg, #ADD8E6cc, #4682B4cc, #87CEEBcc)",
        "table_bg": "linear-gradient(90deg, #E6F0FA, #F5F6F5)",
        "contrast_text": "#1C2526",
    }
}

def apply_theme(theme_name):
    theme = THEMES[theme_name]
    st.markdown(
        f"""
        <style>
            html, body, [data-testid="stAppViewContainer"], .main {{
                font-family: 'Inter', sans-serif;
                background: {theme['background']};
                color: {theme['text_color']};
            }}
            .main .block-container {{
                padding-top: 1rem;
            }}
            h1, h2, h3, h4, h5, h6 {{
                color: {theme['primary_color']} !important;
            }}
            .stButton > button {{
                background-color: {theme['button_bg']};
                color: {theme['contrast_text']};
                border-radius: 10px;
                font-weight: bold;
                transition: background 0.3s;
                padding: 8px 16px;
                border: none;
            }}
            .stButton > button:hover {{
                background-color: {theme['button_hover']};
                color: {theme['contrast_text']};
            }}
            .stTabs [data-testid="stTab"][aria-selected="true"] {{
                background-color: {theme['primary_color']};
                color: {theme['contrast_text']};
                font-weight: bold;
            }}
            .icon {{
                font-size: 1.5em;
                margin-right: 0.3em;
                vertical-align: middle;
                color: {theme['primary_color']};
            }}
            .app-main-width {{
                max-width: 1400px;
                margin-left: auto;
                margin-right: auto;
                padding: 20px;
            }}
            .card {{
                background: {theme['card_bg']};
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 16px;
                box-shadow: 0 3px 6px rgba(0,0,0,0.15);
                color: {theme['text_color']};
            }}
            .weather-header, .iss-header {{
                font-size: 1.6em;
                font-weight: bold;
                margin-bottom: 12px;
                background: {theme['header_gradient']};
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-align: center;
            }}
            .env-summary {{
                font-size: 1.1em;
                background: {theme['weather_gradient']};
                border-radius: 10px;
                padding: 14px;
                margin-bottom: 10px;
                text-align: center;
                color: {theme['contrast_text']};
                border: 1px solid {theme['primary_color']};
            }}
            .table-card {{
                background: {theme['table_bg']};
                border-radius: 10px;
                padding: 10px;
                margin: 10px 0;
                color: {theme['contrast_text']};
            }}
            .iss-info {{
                font-size: 1em;
                background: {theme['card_bg']};
                padding: 12px;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 12px;
                color: {theme['text_color']};
            }}
            .iss-label {{
                color: {theme['secondary_color']};
                font-weight: 600;
            }}
            .eq-item {{
                background: {theme['card_bg']};
                border-left: 4px solid {theme['primary_color']};
                padding: 10px;
                margin-bottom: 8px;
                border-radius: 6px;
                color: {theme['text_color']};
            }}
            .apod-image {{
                max-width: 800px;
                width: 100%;
                height: auto;
                border-radius: 8px;
                margin: 0 auto;
                display: block;
                text-align: center;
            }}
            .slider-container {{
                display: flex;
                align-items: center;
                margin: 10px 0;
            }}
            .slider-label {{
                color: {theme['contrast_text']};
                margin-right: 10px;
                font-size: 0.9em;
            }}
            .switch {{
                position: relative;
                display: inline-block;
                width: 60px;
                height: 34px;
            }}
            .switch input {{
                opacity: 0;
                width: 0;
                height: 0;
            }}
            .slider {{
                position: absolute;
                cursor: pointer;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: {theme['button_bg']};
                transition: .4s;
                border-radius: 34px;
            }}
            .slider:before {{
                position: absolute;
                content: "";
                height: 26px;
                width: 26px;
                left: 4px;
                bottom: 4px;
                background-color: {theme['contrast_text']};
                transition: .4s;
                border-radius: 50%;
            }}
            input:checked + .slider {{
                background-color: {theme['button_hover']};
            }}
            input:checked + .slider:before {{
                transform: translateX(26px);
            }}
            .success-card {{
                background: {theme['card_bg']};
                border-left: 4px solid {theme['primary_color']};
                padding: 10px;
                margin-bottom: 8px;
                border-radius: 6px;
                color: {theme['text_color']};
            }}
        </style>
        """, unsafe_allow_html=True
    )

# --- Session State Initialization ---
def init_session_state():
    defaults = {
        'show_main_app': False,
        'theme': 'Dark',
        'summary_data': {
            "apod": None,
            "mars_rover": None,
            "earth": None,
            "news": None,
            "weather": None,
            "iss_location": None
        },
        'apod_date': datetime.now().date(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- API Helper Functions ---
@st.cache_data(ttl=3600)
def get_apod(date):
    if not is_valid_api_key(API_KEYS["NASA"], "NASA"):
        return None
    try:
        url = f"https://api.nasa.gov/planetary/apod?api_key={API_KEYS['NASA']}&date={date}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        logger.info(f"NASA APOD API called successfully for date: {date}")
        return r.json()
    except requests.RequestException as e:
        logger.error(f"APOD API request failed: {e}")
        return None

@st.cache_data(ttl=3600)
def get_mars_photos(rover, date, camera=None):
    if not is_valid_api_key(API_KEYS["NASA"], "NASA"):
        return None
    try:
        url = f"https://api.nasa.gov/mars-photos/api/v1/rovers/{rover}/photos?api_key={API_KEYS['NASA']}&earth_date={date}"
        if camera:
            url += f"&camera={camera}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        logger.info(f"NASA Mars Photos API called successfully for rover: {rover}, date: {date}")
        return r.json()
    except requests.RequestException as e:
        logger.error(f"Mars Photos API request failed: {e}")
        return None

@st.cache_data(ttl=3600)
def get_weather_forecast(city):
    if not is_valid_api_key(API_KEYS["WEATHER"], "WEATHER"):
        return None
    try:
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEYS['WEATHER']}&units=metric"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        logger.info(f"Weather API called successfully for city: {city}")
        return r.json()
    except requests.RequestException as e:
        logger.error(f"Weather API request failed: {e}")
        return None

@st.cache_data(ttl=10)
def get_iss_location():
    try:
        url = "http://api.open-notify.org/iss-now.json"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
        if "iss_position" in data and "latitude" in data["iss_position"] and "longitude" in data["iss_position"]:
            logger.info("ISS API called successfully")
            return data
        return None
    except requests.RequestException as e:
        logger.error(f"ISS API request failed: {e}")
        return None

@st.cache_data(ttl=3600)
def get_space_news(query):
    if not is_valid_api_key(API_KEYS["NEWS"], "NEWS"):
        return {"articles": []}
    try:
        url = f"https://newsapi.org/v2/everything?q={requests.utils.quote(query)}&language=en&sortBy=publishedAt&apiKey={API_KEYS['NEWS']}&pageSize=20"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        logger.info(f"News API called successfully for query: {query}")
        articles = r.json().get("articles", [])
        keywords = ["space", "nasa", "isro", "esa", "roscosmos", "jaxa", "cnsa", "astronomy", "spacecraft", "rocket", "satellite", "iss"]
        filtered = [
            a for a in articles
            if any(kw in (a.get("title", "") + " " + (a.get("description", "") or "")).lower() for kw in keywords)
        ]
        if len(filtered) < 9:
            filtered = articles[:10]
        return {"articles": filtered[:10]}
    except requests.RequestException as e:
        logger.error(f"News API request failed: {e}")
        return {"articles": []}

@st.cache_data(ttl=3600)
def get_earth_image(lat, lon, date, dim="0.15"):
    if not is_valid_api_key(API_KEYS["NASA"], "NASA"):
        return None, None
    try:
        url = f"https://api.nasa.gov/planetary/earth/imagery?lon={lon}&lat={lat}&date={date}&dim={dim}&api_key={API_KEYS['NASA']}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and r.headers['Content-Type'].startswith('image'):
            logger.info(f"NASA Earth Image API called successfully for lat: {lat}, lon: {lon}")
            return r.content, date
        return None, None
    except requests.RequestException as e:
        logger.error(f"Earth Image API request failed: {e}")
        return None, None

# --- Sentiment Analysis Utilities ---
def analyze_sentiment(text):
    try:
        polarity = TextBlob(text).sentiment.polarity
        return "Positive" if polarity > 0.1 else "Negative" if polarity < -0.1 else "Neutral"
    except Exception:
        return "Neutral"

# --- Geocode and Disaster Demo Functions ---
def geocode_city(city):
    try:
        geolocator = Nominatim(user_agent="astrocastx")
        location = geolocator.geocode(city, language="en", timeout=10)
        if location:
            name = location.address
            name_parts = name.split(", ")
            name_simple = f"{name_parts[0]}, {name_parts[-1]}" if len(name_parts) > 1 else name_parts[0]
            logger.info(f"Geocoded city: {city} to {name_simple}")
            return location.latitude, location.longitude, name_simple
        return None, None, None
    except Exception as e:
        logger.error(f"Geocoding error: {e}")
        return None, None, None

def fetch_aqi(city):
    return np.random.randint(50, 350)

def fetch_pollution_history(city):
    days = pd.date_range(datetime.now() - pd.Timedelta(days=6), periods=7)
    pm25 = np.random.randint(40, 250, size=7)
    return pd.DataFrame({"Date": days, "PM2.5 (μg/m³)": pm25})

def fetch_disasters(lat, lon):
    earthquakes = []
    if lat is None or lon is None:
        return earthquakes
    if (33 < lat < 39 and 135 < lon < 142):
        earthquakes = [
            {"location": "Tokyo, Japan", "magnitude": 6.3, "distance_km": 30, "time": "2 hours ago"},
            {"location": "Osaka, Japan", "magnitude": 5.7, "distance_km": 220, "time": "1 day ago"},
        ]
    elif (32 < lat < 38 and -125 < lon < -114):
        earthquakes = [
            {"location": "San Francisco, USA", "magnitude": 5.8, "distance_km": 45, "time": "5 hours ago"},
        ]
    elif (26 < lat < 32 and 78 < lon < 90):
        earthquakes = [
            {"location": "Kathmandu, Nepal", "magnitude": 6.2, "distance_km": 120, "time": "10 hours ago"},
        ]
    return earthquakes

def fetch_eq_trend(city):
    days = pd.date_range(datetime.now() - pd.Timedelta(days=6), periods=7)
    counts = np.random.randint(0, 2, size=7)
    mags = np.random.uniform(4.5, 6.8, size=7)
    return pd.DataFrame({"Date": days, "Count": counts, "Max_Magnitude": mags})

def fetch_green_initiatives(city):
    return [
        "Citywide Electric Bus roll-out by 2030.",
        "Major afforestation drive: 1 million trees to be planted.",
        "Smart waste management & recycling program.",
        "Subsidies for rooftop solar panels for residents.",
        "Car-free days and expansion of green corridors."
    ]

def aqi_label(aqi):
    if aqi >= 200:
        return "😷 Unhealthy", "red"
    elif aqi >= 100:
        return "🟡 Moderate", "orange"
    else:
        return "🟢 Good", "green"

# --- Welcome Page ---
def welcome_page():
    st.markdown(
        f"""
        <style>
        .welcome-container {{ text-align: center; padding: 40px 0; animation: fadeIn 1s ease-in-out; }}
        .welcome-title {{ font-size: 2.4em; font-weight: bold; background: {THEMES[st.session_state.theme]['header_gradient']};
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .welcome-subtext {{ font-size: 1.1em; max-width: 700px; margin: auto; padding-top: 8px; color: {THEMES[st.session_state.theme]['contrast_text']}; }}
        @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(20px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        </style>
        <div class="welcome-container">
            <img src="https://media.giphy.com/media/xUA7bdpLxQhsSQdyog/giphy.gif" width="180" alt="Rocket Animation"/>
            <h1 class="welcome-title">🚀 AstroCostX</h1>
            <p style='text-align: center; font-size: 0.9em;'>Your Cosmic Journey Awaits</p>
            <p class="welcome-subtext">Discover the universe with real-time ISS tracking, stunning APOD images, space news, and weather forecasts.</p>
        </div>
        """, unsafe_allow_html=True
    )
    if st.button("✨ Start Exploring", use_container_width=True):
        st.session_state.show_main_app = True
        st.rerun()

# --- NASA Explorer Tab ---
def nasa_explorer_tab():
    st.markdown(f'<div class="card"><h2 class="weather-header">🌌 NASA Explorer</h2></div>', unsafe_allow_html=True)
    subtab1, subtab2 = st.tabs(["📷 Astronomy Picture of the Day", "🚗 Mars Rover Explorer"])

    with subtab1:
        today = datetime.now().date()
        apod_date = st.date_input(
            "Select Date", value=st.session_state.apod_date,
            min_value=datetime(1995, 6, 16).date(), max_value=today,
            help="APOD available from June 16, 1995 to today."
        )
        st.session_state.apod_date = apod_date

        with st.spinner("Fetching APOD..."):
            apod_data = get_apod(apod_date.strftime('%Y-%m-%d'))

        if apod_data and apod_data.get("url"):
            title = apod_data.get("title", "Untitled")
            media_type = apod_data.get("media_type", "image")
            explanation = apod_data.get("explanation", "No explanation available.")
            media_url = apod_data.get("url", "")

            st.markdown(f'<div class="card"><h3>{title}</h3>', unsafe_allow_html=True)

            if media_type == "image":
                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <img src="{media_url}" alt="{title}" 
                             style="max-height:600px; width:auto; border-radius:8px;" />
                        <div style="margin-top:5px; font-size:0.9em; color:{THEMES[st.session_state.theme]['contrast_text']};">{title}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif media_type == "video":
                st.video(media_url)
            else:
                st.warning("Unsupported media type.")

            st.markdown("#### 📖 Explanation")
            st.write(explanation)
            st.session_state.summary_data["apod"] = f"{title} ({apod_date}): {explanation[:100]}..."
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("No APOD data found. Check NASA API key or date.")
            st.image("https://apod.nasa.gov/apod/image/2301/NGC2264_HubblePohl_960.jpg", caption="Example APOD (fallback)", use_column_width=False, width=800)
            st.session_state.summary_data["apod"] = "Fallback APOD displayed"

        st.markdown(f'<div class="card"><a href="https://apod.nasa.gov/apod/" target="_blank">🌠 Learn more on NASA APOD</a></div>', unsafe_allow_html=True)

    with subtab2:
        today = datetime.now().date()
        rover = st.selectbox("Select Rover", ["Curiosity", "Opportunity", "Spirit"])
        camera = st.selectbox("Select Camera (optional)", ["All", "FHAZ", "RHAZ", "MAST", "CHEMCAM", "MAHLI", "MARDI", "NAVCAM", "PANCAM", "MINITES"])
        min_dates = {
            "Curiosity": datetime(2012, 8, 6).date(),
            "Opportunity": datetime(2004, 1, 25).date(),
            "Spirit": datetime(2004, 1, 4).date()
        }
        mars_date = st.date_input("Choose Earth Date", today, max_value=today, min_value=min_dates[rover])

        if st.button("📸 Show Mars Photos"):
            data = get_mars_photos(rover.lower(), mars_date.strftime('%Y-%m-%d'), None if camera == "All" else camera)

            if data and data.get('photos'):
                st.success(f"Found {len(data['photos'])} photo(s). Showing top 10.")
                st.markdown('<div class="card">', unsafe_allow_html=True)
                photos = data['photos'][:10]

                for i in range(0, len(photos), 2):
                    cols = st.columns(2)
                    for j in range(2):
                        if i + j < len(photos):
                            photo = photos[i + j]
                            with cols[j]:
                                st.markdown(
                                    f"""
                                    <div style="text-align:center; padding: 10px; border-radius: 8px;">
                                        <img src="{photo['img_src']}" width="100%" style="border-radius: 6px; max-width: 300px;"/>
                                        <div style="font-size:0.85em;color:{THEMES[st.session_state.theme]['contrast_text']};margin-top:8px;">
                                            <strong>Camera:</strong> {photo['camera']['full_name']}<br>
                                            <strong>Rover:</strong> {photo['rover']['name']}<br>
                                            <strong>Date:</strong> {photo['earth_date']}<br>
                                            <strong>Status:</strong> {photo['rover']['status'].capitalize()}
                                        </div>
                                    </div>
                                    """, 
                                    unsafe_allow_html=True
                                )

                st.session_state.summary_data["mars_rover"] = f"{rover}: {len(data['photos'])} photos on {mars_date}"
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("No images found for this rover/camera/date combination.")
                st.session_state.summary_data["mars_rover"] = None

# --- Earth & Satellite Tab ---
def earth_tab():
    st.markdown(f'<div class="card"><h2 class="weather-header">🌍 Environmental Dashboard</h2></div>', unsafe_allow_html=True)
    city = st.text_input(
        "Enter your city or location (e.g. 'London, United Kingdom', 'Delhi, India', 'San Francisco, USA'):",
        "Tokyo",
        help="For best results, enter city and country in English."
    )
    lat, lon, city_display = geocode_city(city)
    if lat is None or lon is None:
        st.error(f"Could not find '{city}'. Try a more specific city/town with country/region in English.")
        return

    tab1, tab2, tab3 = st.tabs(["🌫️ Air Quality", "🌍 Earthquakes", "🌱 Urban Green"])

    with tab1:
        aqi = fetch_aqi(city)
        aqi_status, aqi_color = aqi_label(aqi)
        pollution_df = fetch_pollution_history(city)
        st.markdown(f'<div class="card"><h3>Air Quality in {city_display}</h3><p style="color:{aqi_color};font-size:1.2em">Current AQI: {aqi} {aqi_status}</p></div>', unsafe_allow_html=True)
        fig = px.line(
            pollution_df,
            x="Date",
            y="PM2.5 (μg/m³)",
            title="PM2.5 Trend (Last 7 Days)",
            color_discrete_sequence=[THEMES[st.session_state.theme]["primary_color"]],
            height=300
        )
        fig.add_hline(y=60, line_dash="dash", line_color=THEMES[st.session_state.theme]["secondary_color"], annotation_text="WHO Safety Limit")
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="PM2.5 (μg/m³)",
            template="plotly_white",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("PM2.5 are tiny harmful particles in the air. WHO safe daily average is 60 μg/m³.")
        st.info("**Health Tip:** " + (
            "Avoid outdoor activities, use N95 masks indoors, and keep air purifiers on."
            if aqi >= 200 else
            "Reduce strenuous activities outdoors and monitor sensitive groups."
            if aqi >= 100 else
            "Air is good! Enjoy outdoor activities."
        ))
        st.session_state.summary_data["earth"] = f"{city_display}: AQI {aqi} ({aqi_status})"
        if st.button("AI Ask: Air Quality Summary", key="ai_air"):
            trend_desc = ", ".join(str(x) for x in pollution_df['PM2.5 (μg/m³)'].values)
            prompt = [
                {"role": "system", "content": "You are an expert environmental analyst."},
                {"role": "user", "content":
                    f"""Summarize the air quality for {city_display} for the last 7 days.
                    Current AQI: {aqi} ({aqi_status});
                    PM2.5 levels (last 7 days): {trend_desc}
                    WHO safety limit is 60 μg/m³.
                    Give a short summary, risk level, and advice in clear, plain English."""
                }
            ]
            with st.spinner("AI is analyzing your data..."):
                ai_summary, provider = groq_chat(prompt)
                if not ai_summary.startswith("AI service unavailable"):
                    st.markdown(f'<div class="success-card"><strong>AI Pollution Report:</strong><br>{ai_summary}</div>', unsafe_allow_html=True)
                    st.success("AI Pollution Report generated successfully!")
                else:
                    st.error("Failed to generate AI Pollution Report.")
                    st.info("Fallback AI response: Air quality data unavailable. Ensure proper ventilation and monitor local air quality updates.")

    with tab2:
        earthquakes = fetch_disasters(lat, lon)
        st.markdown(f'<div class="card"><h3>Earthquakes near {city_display}</h3>', unsafe_allow_html=True)
        if earthquakes:
            for eq in earthquakes:
                st.markdown(
                    f"""<div class="eq-item">
                        <strong>{eq['location']}</strong><br>
                        Magnitude: {eq['magnitude']} | Distance: {eq['distance_km']} km | Time: {eq['time']}
                    </div>""", unsafe_allow_html=True
                )
        else:
            st.info(f"No recent earthquakes near {city_display}.")
        st.markdown('</div><div class="card"><h4>Earthquake Activity (Last 7 Days)</h4>', unsafe_allow_html=True)
        eq_trend = fetch_eq_trend(city)
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Bar(
            x=eq_trend["Date"].dt.strftime("%b %d"),
            y=eq_trend["Count"],
            name="Earthquake Count",
            marker_color=THEMES[st.session_state.theme]["primary_color"]
        ))
        fig_eq.add_trace(go.Scatter(
            x=eq_trend["Date"].dt.strftime("%b %d"),
            y=eq_trend["Max_Magnitude"],
            name="Max Magnitude",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color=THEMES[st.session_state.theme]["secondary_color"])
        ))
        fig_eq.update_layout(
            yaxis=dict(title="Earthquake Count", side="left"),
            yaxis2=dict(title="Max Magnitude", overlaying="y", side="right", range=[0, 8]),
            title="Earthquake Count and Max Magnitude",
            template="plotly_white",
            height=300,
            showlegend=True
        )
        st.plotly_chart(fig_eq, use_container_width=True)
        st.caption("Bar: Number of earthquakes per day. Line: Highest magnitude detected.")
        st.markdown('</div>', unsafe_allow_html=True)
        eq_desc = (
            "\n".join([f"{eq['location']} (Magnitude: {eq['magnitude']}, {eq['distance_km']} km away, {eq['time']})" for eq in earthquakes])
            if earthquakes else "No earthquakes near this location recently."
        )
        st.session_state.summary_data["earth"] = f"{city_display}: {eq_desc if earthquakes else 'No recent earthquakes'}"
        if st.button("AI Ask: Earthquake Summary", key="ai_eq"):
            prompt = [
                {"role": "system", "content": "You are an expert disaster risk analyst."},
                {"role": "user", "content":
                    f"""Summarize the recent earthquake activity for {city_display}.
                    Recent earthquakes: {eq_desc}
                    Give a short, plain-English summary and safety tips."""
                }
            ]
            with st.spinner("AI is analyzing your data..."):
                ai_summary, provider = groq_chat(prompt)
                if not ai_summary.startswith("AI service unavailable"):
                    st.markdown(f'<div class="success-card"><strong>AI Earthquake Report:</strong><br>{ai_summary}</div>', unsafe_allow_html=True)
                    st.success("AI Earthquake Report generated successfully!")
                else:
                    st.error("Failed to generate AI Earthquake Report.")
                    st.info("Fallback AI response: No recent earthquakes detected. Prepare an emergency kit and know evacuation routes.")

    with tab3:
        initiatives = fetch_green_initiatives(city)
        st.markdown(f'<div class="card"><h3>Urban Green Initiatives in {city_display}</h3><strong>Major Projects:</strong><ul>{"".join([f"<li>{i}</li>" for i in initiatives])}</ul></div>', unsafe_allow_html=True)
        st.caption("Urban green initiatives help reduce pollution and promote sustainable living.")
        st.session_state.summary_data["earth"] = f"{city_display}: Green initiatives include electric buses, afforestation."
        if st.button("AI Ask: Urban Green Summary", key="ai_green"):
            prompt = [
                {"role": "system", "content": "You are a sustainability and urban planning expert."},
                {"role": "user", "content":
                    f"""Describe the importance and impact of these green initiatives in {city_display}:
                    Initiatives: {"; ".join(initiatives)}
                    Give a motivational, clear summary for city residents."""
                }
            ]
            with st.spinner("AI is analyzing your data..."):
                ai_summary, provider = groq_chat(prompt)
                if not ai_summary.startswith("AI service unavailable"):
                    st.markdown(f'<div class="success-card"><strong>AI Urban Green Report:</strong><br>{ai_summary}</div>', unsafe_allow_html=True)
                    st.success("AI Urban Green Report generated successfully!")
                else:
                    st.error("Failed to generate AI Urban Green Report.")
                    st.info("Fallback AI response: Green initiatives like electric buses and tree planting improve air quality and sustainability.")

# --- Space News Tab ---
def news_tab():
    st.markdown(f'<div class="card"><h2 class="weather-header">📰 Space & Weather News</h2></div>', unsafe_allow_html=True)
    query = st.text_input("🔍 Enter keywords for news", "space OR NASA OR space agency OR weather OR ISRO OR ESA OR JAXA OR Roscosmos OR CNES")
    if st.button("🚀 Get News"):
        news = get_space_news(query)
        if news and news.get("articles"):
            sentiments = {"Positive": 0, "Negative": 0, "Neutral": 0}
            articles_with_sentiment = []
            for article in news["articles"]:
                title = article.get("title", "Untitled")
                url = article.get("url", "#")
                desc = article.get("description") or article.get("content") or ""
                sentiment = analyze_sentiment(desc) if desc.strip() else "Neutral"
                sentiments[sentiment] += 1
                articles_with_sentiment.append({
                    "title": title,
                    "desc": desc[:200] + "..." if desc else "No description.",
                    "url": url,
                    "sentiment": sentiment,
                    "published_at": article.get("publishedAt", "Unknown date")
                })
            st.markdown(f'<div class="card"><h3>Latest News ({len(articles_with_sentiment)} Articles)</h3>', unsafe_allow_html=True)
            for article in articles_with_sentiment:
                sentiment_emoji = "😊" if article["sentiment"] == "Positive" else "😔" if article["sentiment"] == "Negative" else "😐"
                st.markdown(
                    f"""
                    <div class="eq-item">
                        <strong><a href="{article['url']}" target="_blank">{article['title']}</a></strong> {sentiment_emoji}<br>
                        <span style="color:{THEMES[st.session_state.theme]['contrast_text']};font-size:0.85em;">
                            Published: {article['published_at']}<br>
                            {article['desc']}
                        </span>
                    </div>
                    """, unsafe_allow_html=True
                )
            st.session_state.summary_data["news"] = f"{len(news['articles'])} news articles: {sentiments}"
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("**😊 News Sentiment Analysis**")
            sentiment_df = pd.DataFrame(sentiments.items(), columns=["Sentiment", "Count"])
            col_left, col_right = st.columns([1.5, 1])
            with col_left:
                st.markdown('<div class="table-card">', unsafe_allow_html=True)
                st.dataframe(
                    sentiment_df,
                    use_container_width=True,
                    column_config={
                        "Sentiment": st.column_config.TextColumn("Sentiment", width="medium"),
                        "Count": st.column_config.NumberColumn("Count", width="small")
                    }
                )
                st.markdown('</div>', unsafe_allow_html=True)
                st.caption("Sentiment based on article descriptions.")
            
            with col_right:
                if sentiment_df["Count"].sum() > 0:
                    fig = px.pie(
                        sentiment_df,
                        values="Count",
                        names="Sentiment",
                        title="Sentiment Distribution",
                        color="Sentiment",
                        color_discrete_map={
                            "Positive": THEMES[st.session_state.theme]["primary_color"],
                            "Negative": THEMES[st.session_state.theme]["secondary_color"],
                            "Neutral": "#95A5A6"
                        },
                        height=250
                    )
                    fig.update_layout(
                        margin=dict(t=40, b=20, l=20, r=20),
                        title_font_size=14,
                        font=dict(color=THEMES[st.session_state.theme]["contrast_text"])
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Positive: Polarity > 0.1; Negative: Polarity < -0.1; Neutral: Otherwise")
                else:
                    st.warning("No sentiment data available to display.")
            
            if len(articles_with_sentiment) < 9:
                st.warning(f"Only {len(articles_with_sentiment)} articles found. Try broadening the search query to get more results.")
        else:
            st.error("No news found. Check News API key or query.")
            st.session_state.summary_data["news"] = None
            st.markdown("**😊 News Sentiment (Fallback)**")
            fallback_data = pd.DataFrame({
                "Sentiment": ["Positive", "Neutral", "Negative"],
                "Count": [0, 0, 0]
            })
            col_left, col_right = st.columns([1.5, 1])
            with col_left:
                st.markdown('<div class="table-card">', unsafe_allow_html=True)
                st.dataframe(
                    fallback_data,
                    use_container_width=True,
                    column_config={
                        "Sentiment": st.column_config.TextColumn("Sentiment", width="medium"),
                        "Count": st.column_config.NumberColumn("Count", width="small")
                    }
                )
                st.markdown('</div>', unsafe_allow_html=True)
                st.caption("No data due to API failure.")
            with col_right:
                st.warning("No sentiment data available for pie chart due to API failure.")

# --- Weather Tab ---
def weather_tab():
    st.markdown(f'<div class="card"><h2 class="weather-header">☁️ Weather Forecast Dashboard</h2></div>', unsafe_allow_html=True)
    city = st.text_input("🏙️ Enter City Name", "Delhi")
    if st.button("🔍 Show Weather Forecast"):
        forecast = get_weather_forecast(city)
        if forecast and forecast.get("list"):
            data = []
            for item in forecast["list"]:
                dt = datetime.fromtimestamp(item["dt"])
                main = item["main"]
                weather = item["weather"][0]
                rain = item.get("rain", {}).get("3h", 0)
                temp = main.get("temp", 0)
                data.append({
                    "datetime": dt,
                    "temp": temp,
                    "humidity": main.get("humidity", 0),
                    "pressure": main.get("pressure", 0),
                    "weather_main": weather.get("main", ""),
                    "weather_desc": weather.get("description", ""),
                    "wind_speed": item["wind"].get("speed", 0),
                    "rain": rain,
                    "sentiment": analyze_sentiment(weather.get("description", ""))
                })
            df = pd.DataFrame(data)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            colA, colB, colC = st.columns(3)
            with colA:
                st.markdown("**📈 Temperature & Humidity Trend**")
                fig1 = px.line(
                    df,
                    x="datetime",
                    y=["temp", "humidity"],
                    title="Temperature and Humidity Over Time",
                    labels={"value": "Value", "datetime": "Date/Time", "variable": "Metric"},
                    color_discrete_map={
                        "temp": THEMES[st.session_state.theme]["primary_color"],
                        "humidity": THEMES[st.session_state.theme]["secondary_color"]
                    },
                    height=300
                )
                fig1.update_layout(
                    yaxis_title="Temp (°C) / Humidity (%)",
                    template="plotly_white",
                    showlegend=True,
                    hovermode="x unified"
                )
                st.plotly_chart(fig1, use_container_width=True)
                st.caption("Line chart showing temperature (°C) and humidity (%) trends over the forecast period.")
            
            with colB:
                st.markdown("**🌧️ Rainfall Forecast**")
                fig2 = px.bar(
                    df,
                    x="datetime",
                    y="rain",
                    title="Rainfall Over Time",
                    labels={"rain": "Rainfall (mm)", "datetime": "Date/Time"},
                    color_discrete_sequence=[THEMES[st.session_state.theme]["primary_color"]],
                    height=300
                )
                fig2.update_layout(
                    template="plotly_white",
                    showlegend=False,
                    hovermode="x unified"
                )
                st.plotly_chart(fig2, use_container_width=True)
                st.caption("Bar chart showing expected rainfall (mm) every 3 hours.")
            
            with colC:
                st.markdown("**🔵 Temp vs Humidity Correlation**")
                fig3 = px.scatter(
                    df,
                    x="temp",
                    y="humidity",
                    color="temp",
                    title="Temperature vs Humidity",
                    labels={"temp": "Temperature (°C)", "humidity": "Humidity (%)"},
                    color_continuous_scale="RdBu",
                    height=300
                )
                fig3.update_layout(
                    template="plotly_white",
                    showlegend=False,
                    hovermode="closest"
                )
                st.plotly_chart(fig3, use_container_width=True)
                st.caption("Scatter plot showing the relationship between temperature and humidity.")
            
            colD, colE = st.columns(2)
            with colD:
                st.markdown("**📊 Weather Type Frequency**")
                freq = df["weather_main"].value_counts().reset_index()
                freq.columns = ["Weather Type", "Count"]
                fig4 = px.bar(
                    freq,
                    x="Weather Type",
                    y="Count",
                    title="Frequency of Weather Types",
                    color_discrete_sequence=[THEMES[st.session_state.theme]["secondary_color"]],
                    height=300
                )
                fig4.update_layout(
                    template="plotly_white",
                    showlegend=False,
                    hovermode="x unified"
                )
                st.plotly_chart(fig4, use_container_width=True)
                st.markdown('<div class="table-card">', unsafe_allow_html=True)
                st.dataframe(freq, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.caption("Bar chart and table showing the count of different weather types in the forecast.")
            
            with colE:
                st.markdown("**📉 Temperature Distribution**")
                fig5 = px.histogram(
                    df,
                    x="temp",
                    nbins=10,
                    title="Temperature Distribution",
                    labels={"temp": "Temperature (°C)"},
                    color_discrete_sequence=[THEMES[st.session_state.theme]["primary_color"]],
                    height=300
                )
                fig5.update_layout(
                    template="plotly_white",
                    showlegend=False,
                    hovermode="x unified"
                )
                st.plotly_chart(fig5, use_container_width=True)
                st.markdown('<div class="table-card">', unsafe_allow_html=True)
                temp_stats = pd.DataFrame({
                    "Statistic": ["Min", "Max", "Mean", "Std Dev"],
                    "Temp (°C)": [df["temp"].min(), df["temp"].max(), df["temp"].mean(), df["temp"].std()]
                })
                st.dataframe(temp_stats, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.caption("Histogram and table showing the distribution and statistics of temperatures.")
            
            st.markdown("**😊 Weather Sentiment Analysis**")
            sentiment_counts = df["sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]
            st.markdown('<div class="table-card">', unsafe_allow_html=True)
            st.dataframe(sentiment_counts, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption("Sentiment based on weather descriptions (TextBlob polarity).")
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            latest = df.iloc[0]
            current_label = (
                "🌧️ Rainy" if latest["rain"] > 0 else
                "☀️ Sunny" if latest["weather_main"] == "Clear" else
                "☁️ Cloudy" if latest["weather_main"] == "Clouds" else
                "❄️ Cold" if latest["temp"] < 10 else
                "🔥 Hot" if latest["temp"] > 32 else
                "🌨️ Snowy" if latest["weather_main"] == "Snow" else
                "🌫️ Foggy" if latest["weather_main"].lower() in ["mist", "fog", "haze"] else
                "🌥️ Moderate"
            )
            st.markdown(
                f"""<div class="env-summary">
                <b>🌤️ Current Weather in {city}</b><br>
                Status: {current_label}<br>
                Temperature: <b>{latest['temp']:.2f} °C</b><br>
                Humidity: <b>{latest['humidity']}%</b><br>
                Weather: <b>{latest['weather_desc'].capitalize()}</b>
                </div>""", unsafe_allow_html=True
            )
            reviews = []
            if (df["rain"] > 0).sum():
                reviews.append("🌧️ Rainy periods expected.")
            if (df["weather_main"] == "Clear").sum():
                reviews.append("☀️ Sunny/clear periods expected.")
            if (df["weather_main"] == "Clouds").sum():
                reviews.append("☁️ Cloudy periods expected.")
            if (df["temp"] < 10).sum():
                reviews.append("❄️ Cold spells detected.")
            if (df["temp"] > 32).sum():
                reviews.append("🔥 Hot spells detected.")
            if (df["weather_main"] == "Snow").sum():
                reviews.append("🌨️ Possible snow.")
            if (df["weather_main"].str.lower().isin(["mist", "fog", "haze"])).sum():
                reviews.append("🌫️ Misty/Foggy periods expected.")
            avg_humidity = df["humidity"].mean()
            if avg_humidity > 75:
                reviews.append("💧 High average humidity.")
            elif avg_humidity < 40:
                reviews.append("🌵 Low average humidity.")
            review_text = " ".join(reviews)
            st.markdown(
                f"""<div class="env-summary" style="border: 1px solid {THEMES[st.session_state.theme]['secondary_color']};">
                <b>📝 Weather Review</b><br>{review_text}
                </div>""", unsafe_allow_html=True
            )
            st.markdown(
                f"""<div class="env-summary">
                <b>🕒 Forecast Duration:</b><br>
                From <b>{df['datetime'].min().strftime('%Y-%m-%d %H:%M')}</b><br>
                To <b>{df['datetime'].max().strftime('%Y-%m-%d %H:%M')}</b><br><br>
                <b>🔎 Data Points:</b> {len(df)} forecast entries
                </div>""", unsafe_allow_html=True
            )
            st.session_state.summary_data["weather"] = f"{city}: {df['temp'].mean():.1f}°C avg, {df['humidity'].mean():.1f}% humidity"
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("No forecast data found. Check Weather API key or city name.")
            st.session_state.summary_data["weather"] = None
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**📈 Sample Weather Data (Fallback)**")
            fallback_weather = pd.DataFrame({
                "datetime": pd.date_range(datetime.now(), periods=5, freq="3H"),
                "temp": [25.0, 26.0, 24.5, 23.0, 22.5],
                "humidity": [60, 65, 70, 68, 62],
                "weather_main": ["Clear", "Clouds", "Rain", "Clear", "Clouds"],
                "weather_desc": ["clear sky", "scattered clouds", "light rain", "clear sky", "broken clouds"],
                "rain": [0, 0, 0.5, 0, 0],
                "sentiment": ["Positive", "Neutral", "Negative", "Positive", "Neutral"]
            })
            fig1 = px.line(
                fallback_weather,
                x="datetime",
                y=["temp", "humidity"],
                title="Sample Temperature and Humidity",
                labels={"value": "Value", "datetime": "Date/Time", "variable": "Metric"},
                color_discrete_map={
                    "temp": THEMES[st.session_state.theme]["primary_color"],
                    "humidity": THEMES[st.session_state.theme]["secondary_color"]
                },
                height=300
            )
            fig1.update_layout(
                yaxis_title="Temp (°C) / Humidity (%)",
                template="plotly_white",
                showlegend=True,
                hovermode="x unified"
            )
            st.plotly_chart(fig1, use_container_width=True)
            st.caption("Sample data displayed due to API failure.")
            st.markdown('</div>', unsafe_allow_html=True)

# --- AI & Fact Tab ---
def ai_tab():
    st.markdown(f'<div class="card"><h2><span class="icon">🤖</span> AI Summary & Astronomy Fact</h2></div>', unsafe_allow_html=True)
   
    if st.button("Generate Summary & Fact"):
        content = [f"{k}: {v}" for k, v in st.session_state.summary_data.items() if v]
        if not content:
            st.warning("No data available to summarize. Please explore other tabs first (e.g., NASA Explorer, Weather, ISS Tracker).")
            return
        prompt = [
            {"role": "system", "content": "You are an expert in astronomy and data analysis with a passion for making space exciting."},
            {"role": "user", "content": 
                f"""Create a concise, engaging summary for a curious explorer based on this data from AstroCostX:
                {"; ".join(content)}
                Include:
                - A brief highlight from NASA Explorer (e.g., APOD or Mars Rover).
                - Key weather insights (e.g., temperature, conditions).
                - ISS Tracker status (e.g., position, significance).
                - Space News highlights (e.g., number of articles, sentiment).
                - Earth Alert insights (e.g., AQI, earthquakes, green initiatives).
                Keep it under 200 words, fun, and easy to understand."""
            }
        ]
        st.markdown('<div class="card"><h3>AI Cosmic Summary</h3>', unsafe_allow_html=True)
        with st.spinner("Generating AI summary..."):
            summary, provider = groq_chat(prompt)
            if not summary.startswith("AI service unavailable"):
                st.markdown(f'<div class="success-card"><strong>AI Cosmic Summary:</strong><br>{summary}</div>', unsafe_allow_html=True)
                st.success("AI Cosmic Summary generated successfully!")
            else:
                st.error("Failed to generate AI summary.")
                st.info("Fallback AI response: Explore stunning NASA images, track the ISS at 28,000 km/h, check local weather, and stay updated with space news!")
        st.markdown('</div><div class="card"><h3>Random Astronomy Fact</h3>', unsafe_allow_html=True)
        with st.spinner("Generating astronomy fact..."):
            fact_prompt = [
                {"role": "system", "content": "You are an expert in astronomy."},
                {"role": "user", "content": "Give a random, fun, short astronomy fact."}
            ]
            fact, provider = groq_chat(fact_prompt)
            if not fact.startswith("AI service unavailable"):
                st.markdown(f'<div class="success-card"><strong>Astronomy Fact:</strong><br>{fact}</div>', unsafe_allow_html=True)
                st.success("Astronomy Fact generated successfully!")
            else:
                st.error("Failed to generate astronomy fact.")
                st.info("Fallback AI response: The Moon moves away from Earth at about 3.8 cm per year!")
        st.markdown('</div>', unsafe_allow_html=True)

# --- ISS Tracker Tab ---
def iss_tab():
    st.markdown(f'<div class="card"><h2 class="iss-header">🛰️ International Space Station (ISS) Tracker</h2></div>', unsafe_allow_html=True)
    iss_now = get_iss_location()
    if iss_now and iss_now.get("iss_position"):
        iss_lat = float(iss_now['iss_position']['latitude'])
        iss_lon = float(iss_now['iss_position']['longitude'])
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(
                f"""<div class='iss-info'>
                <span class="iss-label">Current ISS Position</span><br>
                <b>Latitude:</b> {iss_lat:.2f}   |   <b>Longitude:</b> {iss_lon:.2f}<br>
                <b>Altitude:</b> ~400 km   |   <b>Speed:</b> ~28,000 km/h
                </div>""", unsafe_allow_html=True
            )
            m = folium.Map(location=[iss_lat, iss_lon], zoom_start=3, tiles="CartoDB positron", width=600, height=400)
            folium.Marker(
                location=[iss_lat, iss_lon],
                tooltip="ISS",
                icon=folium.Icon(color="red", icon="plane")
            ).add_to(m)
            folium.CircleMarker(
                location=[iss_lat, iss_lon], radius=10, color=THEMES[st.session_state.theme]["primary_color"], weight=2, fill=True, fill_opacity=0.2
            ).add_to(m)
            folium_static(m, width=600, height=400)
        with col2:
            st.markdown(
                f"""<div class="card">
                <h3>🚀 ISS Overview</h3>
                <ul style="margin-bottom:0.3em;">
                    <li><b>First Launch:</b> <span style="color:{THEMES[st.session_state.theme]['primary_color']};">20 November 1998</span></li>
                    <li><b>Main Construction:</b> 1998 – 2011</li>
                    <li><b>Orbits Completed:</b> <span style="color:{THEMES[st.session_state.theme]['secondary_color']};">~135,000+</span></li>
                    <li><b>Countries Involved:</b> USA, Russia, Japan, Canada, ESA</li>
                    <li><b>Current Crew:</b> Typically 7 astronauts</li>
                    <li><b>Orbit Period:</b> ~90 minutes</li>
                </ul>
                <b>Mission Highlights:</b>
                <ul>
                    <li>Over <b>3,000</b> scientific experiments</li>
                    <li>Research in microgravity and Earth observation</li>
                    <li>Global collaboration for space exploration</li>
                </ul>
                <b>Learn More:</b> <a href="https://www.nasa.gov/international-space-station/" target="_blank" style="color:{THEMES[st.session_state.theme]['primary_color']};">NASA ISS Page</a>
                </div>""", unsafe_allow_html=True
            )
        st.markdown(
            f"""<div class="card">
            <h3>🧑‍🚀 Fun Facts About the ISS</h3>
            <ul>
                <li><b>Size:</b> Spans 109 meters, roughly a football field.</li>
                <li><b>Speed:</b> Travels at 28,000 km/h, orbiting Earth every 90 minutes.</li>
                <li><b>Visibility:</b> Visible from Earth during dawn or dusk.</li>
                <li><b>Power:</b> Solar panels generate up to 90 kilowatts.</li>
                <li><b>Longevity:</b> Occupied since November 2, 2000.</li>
            </ul>
            <b>Track Live:</b> <a href="https://spotthestation.nasa.gov/tracking_map.cfm" target="_blank" style="color:{THEMES[st.session_state.theme]['primary_color']};">NASA ISS Tracker</a>
            </div>""", unsafe_allow_html=True
        )
        if st.button("AI Ask: ISS Insights"):
            prompt = [
                {"role": "system", "content": "You are an expert in space exploration and astronomy, passionate about making complex information engaging and accessible."},
                {"role": "user", "content":
                    f"""Provide a concise and engaging summary of the International Space Station's current status and its importance for space exploration.
                    Current position: Latitude {iss_lat:.2f}, Longitude {iss_lon:.2f}, Altitude ~400 km, Speed ~28,000 km/h.
                    Include:
                    - A brief description of the ISS's current position and what it might be over (e.g., ocean, continent).
                    - One key scientific contribution of the ISS.
                    - A fun, surprising fact about life on the ISS.
                    Keep it clear, exciting, and under 150 words."""
                }
            ]
            with st.spinner("Generating AI ISS Insights..."):
                ai_summary, provider = groq_chat(prompt)
                if not ai_summary.startswith("AI service unavailable"):
                    st.markdown(f'<div class="success-card"><strong>AI ISS Insights:</strong><br>{ai_summary}</div>', unsafe_allow_html=True)
                    st.success("AI ISS Insights generated successfully!")
                else:
                    st.error("Failed to generate AI ISS Insights.")
                    st.info("Fallback AI response: The ISS orbits Earth at 28,000 km/h, conducting vital microgravity experiments. Fun fact: Astronauts see 16 sunrises daily!")
        st.session_state.summary_data["iss_location"] = f"ISS at ({iss_lat:.2f}, {iss_lon:.2f})"
    else:
        st.error("ISS location data unavailable.")
        st.session_state.summary_data["iss_location"] = None
        st.markdown('<div class="card">', unsafe_allow_html=True)
        m = folium.Map(location=[0, 0], zoom_start=2, tiles="CartoDB positron", width=600, height=400)
        folium.Marker(
            location=[0, 0],
            tooltip="Sample ISS Location",
            icon=folium.Icon(color="red", icon="plane")
        ).add_to(m)
        folium_static(m, width=600, height=400)
        st.caption("Sample ISS location shown due to API failure.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- Main App ---
def main_app():
    st.markdown('<div class="app-main-width">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="slider-container">
            <span class="slider-label">Switch to {'Light' if st.session_state.theme == 'Dark' else 'Dark'} Theme</span>
            <label class="switch">
                <input type="checkbox" {'checked' if st.session_state.theme == 'Light' else ''} id="theme-toggle">
                <span class="slider"></span>
            </label>
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown(
        """
        <script>
        document.getElementById('theme-toggle').addEventListener('change', function() {
            const theme = this.checked ? 'Light' : 'Dark';
            fetch('/_stcore/update-session-state', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({theme: theme})
            }).then(() => window.location.reload());
        });
        </script>
        """, unsafe_allow_html=True
    )
    apply_theme(st.session_state.theme)
    tabs = st.tabs([
        "🌌 NASA Explorer",
        "🌍 Earth Alert",
        "📰 Space News",
        "☁️ Weather",
        "🤖 AI & Fact",
        "🛰️ ISS Tracker"
    ])
    with tabs[0]:
        nasa_explorer_tab()
    with tabs[1]:
        earth_tab()
    with tabs[2]:
        news_tab()
    with tabs[3]:
        weather_tab()
    with tabs[4]:
        ai_tab()
    with tabs[5]:
        iss_tab()
    st.markdown(
        f"""
        <style>
        .footer-custom {{
            text-align: center;
            font-size: 0.85em;
            color: {THEMES[st.session_state.theme]['contrast_text']};
            margin-top: 50px;
            padding: 10px 0;
            border-top: 1px solid #ccc;
        }}
        .footer-custom a {{
            color: {THEMES[st.session_state.theme]['contrast_text']};
            margin: 0 8px;
            text-decoration: none;
        }}
        .footer-custom img {{
            vertical-align: middle;
            margin: 0 5px;
        }}
        </style>

        <div class="footer-custom">
            Powered by AI | Built with Streamlit | © 2025 <strong>AstroCostX</strong><br>
            Made by <strong>Shivendra Kumar Pandey</strong> | 📧 
            <a href="mailto:pandeyshr2006@gmail.com">pandeyshr2006@gmail.com</a><br>
            <a href="https://www.linkedin.com/in/shivendra-kumar-pandey" target="_blank" title="LinkedIn">
                <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="20">
            </a>
            <a href="https://github.com/shivendrapandey" target="_blank" title="GitHub">
                <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="20">
            </a>
            <a href="https://x.com/shivendrapandey" target="_blank" title="X (Twitter)">
                <img src="https://abs.twimg.com/favicons/twitter.2.ico" width="20">
            </a>
        </div>
        """, 
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

# --- Entrypoint ---
if __name__ == "__main__":
    if not st.session_state.show_main_app:
        welcome_page()
    else:
        main_app()