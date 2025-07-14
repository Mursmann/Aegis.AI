# Aegis.AI – Neural Protection Engine

Aegis.AI is an intelligent, adaptive host-based intrusion detection system (HIDS) designed to protect both servers and personal computers from abnormal or malicious behavior using machine learning.

## 🔒 Project Summary

**Aegis.AI** leverages real-time system behavior monitoring and machine learning models to detect potential threats such as:

- Unusual CPU or memory spikes  
- Sudden increases in the number of processes  
- Abnormal file system activity (e.g., mass file creation)  
- Unexpected network connections  

Whether you're protecting a cloud server or a local desktop, Aegis.AI adapts to the behavior profile of the host to distinguish between normal and potentially dangerous activity.

## 🧠 Key Features

- ✅ Real-time behavioral data logging (CPU, RAM, processes, network, filesystem)
- ✅ ML-based classification engine trained on normal and abnormal usage patterns
- ✅ Supports both anomaly-based and signature-based detection logic
- ✅ Modular and extendable architecture (can add new sensors or detectors)
- ✅ Lightweight agent design optimized for low system impact
- ✅ Customizable risk sensitivity modes: Light, Balanced, Aggressive

## 📊 Architecture Overview

```
[ Host System ]
     ↓
[ Behavioral Logger (psutil, watchdog) ]
     ↓
[ Feature Engineering + Labeling ]
     ↓
[ ML Model (Random Forest / Neural Network) ]
     ↓
[ Threat Detection Engine ]
     ↓
[ GUI Dashboard / Alerting System ]
```

## 🧪 Datasets

Aegis.AI supports both:
- Public datasets for initial training (e.g., UNSW-NB15, ADFA-LD)
- Real-time local logs collected via built-in sensors

Custom simulation modules are planned to allow safe testing of malicious patterns on isolated machines.

## 💻 Compatibility

- OS: Linux, Windows
- Python 3.8+
- Lightweight enough for personal laptops and IoT devices  
- Can be deployed as a background service or standalone executable

## 🚀 Roadmap

- [x] Data collection module  
- [x] Feature extraction & labeling  
- [x] Initial model training (Random Forest)  
- [ ] Neural network integration  
- [ ] Interface for end users (GUI/CLI)  
- [ ] Cross-platform installer and daemon service  
- [ ] Live response module (auto-quarantine or alerts)  

## 📁 Structure (example folders)

```
/data         → Raw & processed behavior logs  
/model        → Trained ML models  
/src          → Source code (logger, trainer, detection engine)  
/interface    → UI components (future)  
/experiments  → Notebooks, test results  
```

## 🤖 Technologies Used

- Python 3 (main language)  
- psutil, watchdog – for system behavior monitoring  
- scikit-learn, xgboost, pandas – for modeling  
- matplotlib, seaborn – for visual analytics  
- Optional: Tkinter or Streamlit – for GUI

## 👨‍💻 Author & Collaboration

Created by Woldemar Mursmann  
Contributions and testing are welcome — please open issues or pull requests.

## 📜 License

This project is open-source under the **M_W_PROGECT**.
