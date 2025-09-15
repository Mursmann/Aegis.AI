# Aegis.AI – Neural Protection Engine
![Logo](NewLOGO.jpg)

🛡️ Aegis.AI
Aegis.AI is a cutting-edge Host-based Intrusion Detection System (HIDS) leveraging machine learning and artificial intelligence to detect threats in real time. The project focuses on behavioral analysis of files, processes, and system resources to protect against malware, including cryptominers, ransomware, and sophisticated attacks. Aegis.AI combines lightweight ML models for rapid screening with powerful transformers for in-depth analysis of suspicious entities.
🤖👾👾👾🤖
📖 About the Project
Aegis.AI is an open-source cybersecurity solution designed to monitor and protect hosts from threats. Key features include:
File Analysis: Detection of malicious PE files (EXE, DLL) and scripts (PowerShell, Python) using XGBoost and transformer models.
CPU Monitoring: Identification of anomalous CPU usage patterns, characteristic of cryptominers or other suspicious processes.
Two-Stage Detection: Fast ML-based screening (XGBoost, Random Forest) for initial classification, followed by deep analysis with Hugging Face transformers for confirmed threats.
Modular Architecture: Lightweight agent with real-time capabilities, integrated with psutil and watchdog.
Aegis.AI leverages datasets like UNSW-NB15 and ADFA-LD and integrates with state-of-the-art AI models to enhance detection accuracy.
🚀 Features
Real-Time Monitoring: Tracks file system changes and process activity using watchdog and psutil.
Behavioral Analysis: Detects anomalies in file structures (e.g., PE entropy, imports) and process behavior (e.g., CPU/memory usage, network connections).
ML-Driven Detection: Uses XGBoost and Random Forest for efficient, low-resource initial threat screening.
Transformer Integration: Employs Hugging Face models (e.g., GeLECTRA, MalBERT) for precise analysis of suspicious files and processes.
Extensible Design: Modular codebase supports easy integration of new models and features.

📈 Usage
Aegis.AI operates as a lightweight HIDS agent:
File Monitoring: Scans new or modified files (e.g., EXE, DLL, PowerShell scripts) for malicious patterns.
Process Monitoring: Detects processes with high CPU usage (>80%) or suspicious behavior (e.g., excessive network connections).
Two-Stage Pipeline:
ML Screening: XGBoost evaluates file/process features for rapid threat detection.
Transformer Analysis: Suspicious entities are analyzed by transformer models for detailed confirmation.
Example workflow in detection_engine.py:
from src.hf_detector import HFDetector
from src.detection_engine import ml_screening

detector = HFDetector(model_name="deepset/gelectra-base-malware")
for proc in psutil.process_iter(['pid', 'cpu_percent']):
    if ml_screening(extract_process_features(proc.info['pid'])):
        label, score = detector.scan_process(proc.info['pid'])
        if label == "MALWARE":
            print(f"Threat confirmed: Process {proc.info['pid']} ({score})")
📅 Roadmap
[x] Implement ML-based anomaly detection with XGBoost and Random Forest.
[x] Integrate file and process monitoring with watchdog and psutil.
[ ] Add transformer-based deep analysis for suspicious files/processes (Hugging Face).
[ ] Develop a GUI for real-time threat visualization and alerts.
[ ] Introduce live response capabilities (e.g., process termination, file quarantine).
[ ] Support additional datasets and fine-tuning for custom environments.
[ ] Optimize performance for low-resource systems.

📚 Resources
Datasets: UNSW-NB15, ADFA-LD, Ember, VirusShare (for PE files).
Models: Hugging Face (deepset/gelectra-base-malware, WhiteRabbitNeo/neo-malware-analyzer).
Tools: pefile for PE analysis, psutil for system monitoring, watchdog for file events.
📬 Contact
Have questions or ideas? Reach out via GitHub Issues or connect with the community on X.
📜 License
This project is licensed under the MIT License. See LICENSE for details.
## 📜 License

This project is open-source under the **M_W_PROJECT**.
