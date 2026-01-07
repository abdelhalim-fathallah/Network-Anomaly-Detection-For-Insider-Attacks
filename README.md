# 🛡️ Network Anomaly Detection for Insider Attacks
### HIMIT Graduation Project | Class of 2025

---

## 📝 Overview
This project is a comprehensive cybersecurity solution developed by our team at **HIMIT**. It focuses on the critical challenge of detecting **Insider Attacks**—malicious activities performed by authorized users within an organization. Unlike external threats, insider attacks are notoriously difficult to identify using traditional security measures.

Our system leverages **Machine Learning and Statistical Analysis** to monitor network traffic, extract behavioral features, and identify subtle anomalies that indicate potential security breaches.

## 🎯 Objectives
* **Anomaly Detection:** Identify deviations in network traffic indicative of insider threats.
* **Automation:** Provide a scalable and automated monitoring solution for modern organizations.
* **Real-time Alerting:** Generate immediate notifications for suspicious activities to mitigate data exfiltration.



## Technical Highlights
The system is built on a robust **End-to-End Pipeline**:
* **Ensemble Intelligence:** Uses a **Weighted Ensemble Model** combining `Random Forest`, `XGBoost`, and `Gradient Boosting` to achieve near-perfect detection rates.
* **Feature Engineering:** Analysis of 41 high-dimensional network features based on the **NSL-KDD** benchmark dataset.
* **Interactive Dashboard:** A professional UI built with `React.js` and `Tailwind CSS`, featuring real-time data visualization via `Recharts`.
* **Production Ready:** Fully containerized using `Docker` and ready for cloud deployment via `Render`.

## Performance Metrics
The system delivers state-of-the-art results:
* **Accuracy:** 99.92%
* **F1-Score:** 99.91%
* **Precision/Recall:** 99.92% / 99.91%
* **AUC-ROC:** 100% (demonstrating perfect class separation).

## 📁Project Structure
```
network-anomaly-detection/
├── backend/
│   ├── data/           # NSL-KDD Dataset files
|   ├── models/         # Ensemble & Autoencoder models (Active)
│   ├── app.py          # Flask API Gateway
│   └── train_models.py # Model training & evaluation logic
|
├── frontend/
│   ├── src/            # React components & Dashboard logic
│   ├── public/         # Static assets and index.html
│   └── package.json    # Frontend dependencies
├── Dockerfile          # Backend containerization
├── start-project.ps1   # One-click automation script
└── README.md           # Documentation
```


## Getting Started
Prerequisites
Python 3.10+
Node.js & npm
PowerShell (for automation script)

## Installation & Execution
Automated Start (Recommended):
```
.\start-project.ps1
```
## Manual Setup:
```
Backend: cd backend && pip install -r requirements.txt && python app.py
Frontend: cd frontend && npm install && npm start
```
## Team Members
```
• Abdelhalim Mohsen Fathallah <br />
• Sameh Mahmoud El-Gebally 
• Asmaa Ibrahim lila 
• Mohamed Ali Abdulmuti 
• Mahmoud Hossam El-dein El-Gohary 
```

### References & Technologies
AI/ML: Scikit-learn, XGBoost, Pandas, NumPy.
Web: Flask (Backend), React (Frontend).
Dataset: NSL-KDD (Improved version of KDD Cup 99).
Analysis: PCAP analysis and behavioral modeling.



**Developed for educational purposes as a graduation project demonstration.**