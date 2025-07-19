# PestScanner4-Classification-app

to deply like in cloud :
ngrok http 5000


## Setup Instructions

### Raspberry Pi Setup

1. **Hardware Requirements**:
   - Raspberry Pi (any model with UART)
   - GPS module (NMEA-compatible)

2. **Software Dependencies**:
```bash
   gps_tracking_project/
├── raspberry_pi/
│   ├── gps_to_firebase.py          # Python script to read GPS and send to Firebase
│   └── your-project-id-firebase-adminsdk-xxxxx-xxxxxxxxxx.json  # Firebase service account key
├── web_app/
│   └── index.html                  # Web interface to display real-time tracking
└── README.md                       # Project documentation (optional)
 ```
here it's the link of the app live you can test it: 
```bash
https://pestscanner4-classification-app-for-citron-disease.streamlit.app/
 ```
and here it's the link of the website:
```bash
https://v0-pest-scanner-website-design-nnxxo79fx.vercel.app/
 ```
