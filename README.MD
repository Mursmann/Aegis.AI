# 🛡️ Aegis.AI – Neural Protection Engine
![Logo](NewLOGO.jpg)

---

## 🤖 Overview
Aegis.AI is a cutting-edge Host-based Intrusion Detection System (HIDS) leveraging machine learning and artificial intelligence to detect threats in real time.  
It focuses on behavioral analysis of files, processes, and system resources to protect against malware such as cryptominers, ransomware, and advanced persistent threats (APT).  

Aegis.AI combines lightweight ML models for rapid screening with powerful transformers for in-depth analysis of suspicious entities.  

---

## 📖 About the Project
Key capabilities include:

- 📂 File Analysis  
  Detection of malicious PE files (EXE, DLL) and scripts (PowerShell, Python) using XGBoost and transformer models.  

- 🖥️ CPU Monitoring  
  Identifies anomalous CPU usage patterns, typical of cryptominers or hidden malicious processes.  

- ⚡ Two-Stage Detection  
  1. Fast ML-based screening (XGBoost, Random Forest).  
  2. Deep transformer analysis (Hugging Face models) for high-confidence results.  

- 🧩 Modular Architecture  
  Lightweight agent built with psutil and watchdog, ready for extension with new models.  

Datasets used: UNSW-NB15, ADFA-LD.  
Models: Hugging Face transformers such as GeLECTRA and MalBERT.  

---

## 🚀 Features
- 🔍 Real-Time Monitoring – File system & process tracking.  
- 📊 Behavioral Analysis – PE entropy, imports, process CPU/memory/network patterns.  
- 🤖 ML-Driven Detection – XGBoost & Random Forest classifiers.  
- 🧠 Transformer Integration – Hugging Face models for deep detection.  
- 🔧 Extensible Design – Modular codebase for new models/features.  

---

## 📈 Usage
Run as a lightweight HIDS agent:

- 🗂️ File Monitoring – Scans new/modified files (EXE, DLL, PowerShell).  
- 🖥️ Process Monitoring – Detects suspicious processes (e.g., CPU > 80%).  
- 🔄 Two-Stage Pipeline:  
  1. ML screening (fast classification).  
  2. Transformer analysis (deep confirmation).  

---

📅 Roadmap

[x] Implement ML anomaly detection (XGBoost, Random Forest).

[x] File & process monitoring (watchdog, psutil).

[ ] Transformer-based deep analysis (Hugging Face).

[ ] GUI for visualization & alerts.

[ ] Live response (terminate process, quarantine file).

[ ] Support additional datasets & fine-tuning.

[ ] Optimize for low-resource systems.



---

📚 Resources

Datasets: UNSW-NB15, ADFA-LD, Ember, VirusShare.

Models: Hugging Face – deepset/gelectra-base-malware, WhiteRabbitNeo/neo-malware-analyzer.

Tools: pefile, psutil, watchdog.



---

