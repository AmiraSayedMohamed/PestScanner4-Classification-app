# 🌱 PestScanner Disease Classification App
![App Screenshot](https://github.com/AmiraSayedMohamed/PestScanner4-Classification-app/blob/master/1.jpg)
<p align="center">
 <p align="center">
  <img src="https://github.com/AmiraSayedMohamed/PestScanner4-Classification-app/blob/master/1.jpg?raw=true" width="200" style="margin: 10px;">
  <img src="https://github.com/AmiraSayedMohamed/PestScanner4-Classification-app/blob/master/2.jpg?raw=true" width="200" style="margin: 10px;"><br>
  <img src="https://github.com/AmiraSayedMohamed/PestScanner4-Classification-app/blob/master/3.jpg?raw=true" width="200" style="margin: 10px;">
  <img src="https://github.com/AmiraSayedMohamed/PestScanner4-Classification-app/blob/master/6.jpg?raw=true" width="200" style="margin: 10px;">
</p>

<p align="center">
  <img src="https://github.com/AmiraSayedMohamed/PestScanner4-Classification-app/blob/master/4.jpg?raw=true" width="200" style="margin: 10px;">
  <img src="https://github.com/AmiraSayedMohamed/PestScanner4-Classification-app/blob/master/5.jpg?raw=true" width="200" style="margin: 10px;">
  <img src="https://github.com/AmiraSayedMohamed/PestScanner4-Classification-app/blob/master/7.jpg?raw=true" width="200" style="margin: 10px;"><br>
  <img src="https://github.com/AmiraSayedMohamed/PestScanner4-Classification-app/blob/master/8.jpg?raw=true" width="200" style="margin: 10px;">
</p>



A powerful AI-powered web application for classifying citrus leaf diseases, visualizing pest outbreak maps, and analyzing weather-related disease risks — tailored for Egyptian farmers and agricultural experts.
to deply like in cloud :
ngrok http 5000

---

## 📌 Features

- 🔍 **Image-Based Disease Detection** using TensorFlow Lite
- 🧠 **Weather-Integrated Risk Forecasting** via OpenWeather API
- 🗺️ **Interactive Pest Monitoring Map** using Folium
- 🧪 **Localized Treatment Recommendations** (Chemical, Organic, and Cultural)
- 🤖 **Agricultural AI Chatbot** via OpenRouter for expert advice
- 🧭 **Geolocation-Aware City Selection** (covering 40+ Egyptian cities)
- 📊 **Charts and Reports** for visualizing disease types and severity trends
- **Recommendation System** on how to treat teh disease
- **Technical Support to answer any question you want inside the part ask and expert**

---

## 🖼️ Example Use Case

1. Upload a **citrus leaf image** (JPG/PNG)
2. Get a **disease diagnosis** (Black Spot or Citrus Canker)
3. View:
   - Recommended treatments
   - Disease probability distribution
   - Weather-based spread forecast
   - Interactive map of reported outbreaks in your area

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/pestscanner.git
cd pestscanner

 ```
here it's the link of the app live you can test it: 
```bash
https://pestscanner4-classification-app-for-citron-disease.streamlit.app/
 ```
and here it's the link of the website:
```bash
https://v0-pest-scanner-website-design-nnxxo79fx.vercel.app/
 ```
to test the app you can upload this two pictures:
- https://github.com/AmiraSayedMohamed/PestScanner4-Classification-app/blob/master/black-spot-disease-pic.jpg
- https://github.com/AmiraSayedMohamed/PestScanner4-Classification-app/blob/master/CitrucCanker-Disease-pic.jpg
-----------
### Structe of the repo:
- 1.jpg to 8.jpg is the pictures of the app
- CitrucCanker-Disease-pic.jpg  and  black-spot-disease-pic.jpg   are  picture you can use to test the functionality of the app
- RaspberryPi-local-detect-count- telegramm , is code that you can run on yourcomputer local to dectect the disease real time using custom data
- plant_disease_classifier-quant.tf ---- i have make quantization to my classification model to reduce it's size

