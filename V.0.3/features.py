import os
import ember
import numpy as np
import psutil
import lief
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def extract_features_pe(file_path):
    try:
        logging.info(f"Checking PE file: {file_path}")
        if not lief.is_pe(file_path):
            logging.warning(f"{file_path} is not a valid PE file")
            return None
        with open(file_path, "rb") as f:
            binary = f.read()
        extractor = ember.PEFeatureExtractor()
        features = np.array(extractor.feature_vector(binary), dtype=np.float32)
        logging.info(f"Features successfully extracted from {file_path}")
        return features
    except Exception as e:
        logging.error(f"Error extracting features from {file_path}: {e}")
        return None

def analyze_non_pe_file(file_path):
    try:
        if not os.access(file_path, os.R_OK):
            logging.warning(f"No access rights to file {file_path}")
            return 0.0
        suspicious_keywords = [
            "mining", "stratum", "pyminer", "cryptocurrency", "wallet", "pool",
            "import os", "subprocess.run", "socket.connect", "requests.post"
        ]
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().lower()
        if any(keyword in content for keyword in suspicious_keywords):
            logging.warning(f"⚠️ Suspicious file (keywords found): {file_path}")
            return 1.0
        logging.info(f"✅ File is safe (heuristics): {file_path}")
        return 0.0
    except Exception as e:
        logging.error(f"Error analyzing file {file_path}: {e}")
        return 0.0

def collect_behavior_features(proc):
    try:
        cpu_percent = proc.cpu_percent(interval=0.1)
        memory_percent = proc.memory_percent()
        net_connections = len(proc.net_connections()) if hasattr(proc, 'net_connections') else 0
        open_files = len(proc.open_files()) if hasattr(proc, 'open_files') else 0
        threads = proc.num_threads()
        return [cpu_percent / 100.0, memory_percent / 100.0, net_connections / 10.0, open_files / 100.0, threads / 10.0]
    except Exception as e:
        suppressed = ['svchost.exe', 'WUDFHost.exe']
        name = proc.info.get('name', '').lower()
        if name not in suppressed:
            logging.error(f"Error collecting process features {proc.info['name']}: {e}")
        return [0.0, 0.0, 0.0, 0.0, 0.0]