# 🌌 AstroCast

**AstroCast** is an interactive Streamlit dashboard that brings together the wonders of space, live weather data, cutting-edge AI summaries, and more, all in an elegant, visual, and user-friendly web application.

---

## 🚀 What is AstroCast?

**AstroCast** is your all-in-one portal to discover the cosmos and our planet with real-time data, news, and imagery powered by NASA, OpenWeather, NewsAPI, and modern AI. Whether you're a space enthusiast, student, educator, or just curious, AstroCast lets you:

- 📷 Explore NASA's Astronomy Picture of the Day in stunning high resolution.
- 🚗 Browse the latest photos beamed back by Mars Rovers (Curiosity, Opportunity, Spirit).
- 🌍 See satellite imagery of any location on Earth.
- 📰 Stay updated with the latest space and weather news, automatically analyzed for sentiment.
- ☁️ Visualize weather forecasts for any city, with insightful charts and summaries.
- 🛰️ Track the International Space Station live on an interactive world map.
- 🤖 Get AI-generated summaries and fun astronomy facts at any time.

---

## 🖥️ Features Explained

### 1. **Astronomy Picture of the Day (APOD)**
- View NASA's daily photo or video from space, complete with NASA's official description.
- Select any date since June 16, 1995, to see that day's APOD.
- Images are displayed large and centered for maximum visual impact.

### 2. **Mars Rover Explorer**
- Select a Mars Rover (Curiosity, Opportunity, Spirit) and view photos taken by their onboard cameras.
- Filter by date and camera type.
- Each photo includes mission data: rover name, camera, date, landing/launch, and status.

### 3. **Earth & Satellite View**
- Input any latitude and longitude to see recent NASA satellite imagery of Earth.
- Great for seeing weather systems, geography, or satellite views of famous locations.

### 4. **Space & Weather News**
- Aggregates news from around the world related to space agencies, missions, weather, and astronomy.
- News articles are auto-analyzed for sentiment (positive, negative, neutral) with charts and tables.
- Clickable headlines and short summaries for quick browsing.

### 5. **Weather Forecast Dashboard**
- Enter any city to see a 5-day forecast from OpenWeather.
- Interactive charts: temperature trends, humidity, rainfall, weather type frequency, and more.
- Instant summaries of weather patterns and climate insights.

### 6. **International Space Station (ISS) Tracker**
- See the ISS's exact position live on a world map (updated in real time).
- Includes ISS mission history, fun facts, and direct links to NASA's official tracker.
- See stats like launch date, mission highlights, and ongoing research (no astronaut list shown for privacy).

### 7. **AI Summary & Astronomy Fact**
- With one click, get a fun, AI-generated summary of your current AstroCast session.
- Ask for a random astronomy fact to learn something new every time.

---

## 📦 Installation & Usage

### 1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/astrocast.git
cd astrocast
```

### 2. **Install Dependencies**

All required libraries are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. **API Keys Needed**

**You must obtain API keys for the following:**
- NASA API: [https://api.nasa.gov/](https://api.nasa.gov/)
- NewsAPI: [https://newsapi.org/](https://newsapi.org/)
- OpenWeather: [https://openweathermap.org/api](https://openweathermap.org/api)
- GROQ API: [https://console.groq.com/](https://console.groq.com/)
- Google Generative AI: [https://ai.google.dev/](https://ai.google.dev/)

Add these keys directly in the top of `app.py` or use environment variables for security.

### 4. **Run AstroCast**

```bash
streamlit run app.py
```

Go to the provided local URL in your web browser to enjoy AstroCast!

---

## 🧑‍💻 How It Works

- **Streamlit** powers the fast, interactive, and responsive web UI.
- **NASA APIs** provide real-time APOD images, Mars rover photos, and Earth satellite imagery.
- **OpenWeather** delivers detailed, up-to-date weather data and forecasts.
- **NewsAPI** fetches the latest news, which is processed using Python's `TextBlob` for sentiment analysis.
- **Folium** and **streamlit-folium** render beautiful, interactive maps for ISS tracking and location-based data.
- **GROQ** and **Google Generative AI** generate friendly summaries and fun facts live in-app.
- **Pandas & Matplotlib** power all data analysis and charting, so you can visualize trends instantly.

---

## 🎨 User Experience

- **Modern, wide layout:** Designed for desktops and large screens, but responsive to browser resizing.
- **Beautiful section headings:** Easy to navigate and visually pleasing.
- **Large APOD images:** Enjoy NASA's space photography in all its glory.
- **No astronaut privacy issue:** ISS tab does not display current crew members.
- **No theme toggle:** Uses a clean, consistent look for clarity and focus.

---

## 🛠️ File Structure

- **app.py** – The single main app file (just run with Streamlit).
- **requirements.txt** – All dependencies, ready to install.
- **README.md** – This documentation.

---

## 💡 Example Use Cases

- **Students & Teachers:** Explore current events in space, visualize weather patterns, and inspire learning.
- **Space Fans:** Stay on top of missions, launches, and cosmic discoveries.
- **Educators:** Use as a live demo or interactive board in classroom settings.
- **Curious Minds:** Check out the ISS, see what’s happening on Mars, or just browse the universe!

---

## 🙏 Credits

AstroCast uses data and imagery courtesy of:
- [NASA Open APIs](https://api.nasa.gov/)
- [OpenWeather](https://openweathermap.org/)
- [NewsAPI](https://newsapi.org/)
- [GROQ AI](https://groq.com/)
- [Google Generative AI](https://ai.google.dev/)
- All open source contributors in the Python and Streamlit communities.

---

## 📃 License

MIT License. See [LICENSE](LICENSE) for full details.

---

## 🤝 Contributing

Issues, suggestions, and pull requests are welcome. Please open an issue if you find a bug or have an idea to improve AstroCast!

---

## 📸 Screenshots

> Add your screenshots of AstroCast here!

---