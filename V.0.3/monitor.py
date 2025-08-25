import os
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import tensorflow as tf
import joblib
import psutil
import time
import logging
from features import extract_features_pe, analyze_non_pe_file, collect_behavior_features
from utils import TransformerLayer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def monitor_system(directory):
    logging.info("🚀 Starting monitoring...")
    try:
        model_lgb = joblib.load("ember_model_lgb.lgb")
        model_xgb = xgb.Booster()
        model_xgb.load_model("xgb_model.json")
        model_meta = tf.keras.models.load_model("meta_model.h5")
        transformer_model = tf.keras.models.load_model("transformer_model.h5", custom_objects={'TransformerLayer': TransformerLayer})
        logging.info("✅ Models loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        return

    MONITOR_DIR = directory if directory else os.getcwd()
    checked_files = set()
    MINING_POOLS = ["pool.minergate.com", "xmr.pool.minergate.com", "pool.nicehash.com"]
    PE_EXTENSIONS = [".exe", ".dll"]
    current_script = "main.py"

    while True:
        try:
            files = [f for f in os.listdir(MONITOR_DIR) if f not in checked_files and f != current_script]
            if not files:
                logging.info(f"No new files to check in directory {MONITOR_DIR}")
            for file in files:
                file_path = os.path.join(MONITOR_DIR, file)
                features = []
                if any(file.lower().endswith(ext) for ext in PE_EXTENSIONS):
                    pe_features = extract_features_pe(file_path)
                    if pe_features is not None:
                        pe_features = np.array(pe_features).reshape(1, -1)
                        y_pred_lgb = model_lgb.predict(pe_features)
                        dtest_xgb = xgb.DMatrix(pe_features)
                        y_pred_xgb = model_xgb.predict(dtest_xgb)
                        features = [y_pred_lgb[0], y_pred_xgb[0], 0.0, 0.0, 0.0]
                    else:
                        features = [0.0, 0.0, 0.0, 0.0, 0.0]
                else:
                    heuristic_score = analyze_non_pe_file(file_path)
                    features = [0.0, 0.0, heuristic_score, 0.0, 0.0]
                
                if features:
                    logging.info(f"Transformer input features (file {file}): {features}")
                    transformer_input = np.array(features).reshape(1, 1, -1)
                    y_pred_transformer = transformer_model.predict(transformer_input, verbose=0)
                    if y_pred_transformer[0] > 0.5:
                        logging.warning(f"⚠️ Malicious file detected (transformer): {file_path}")
                    else:
                        logging.info(f"✅ File is safe (transformer): {file_path}")
                checked_files.add(file)

            for proc in psutil.process_iter(['name', 'exe', 'cpu_percent']):
                try:
                    behavior_features = collect_behavior_features(proc)
                    if proc.info['cpu_percent'] > 80:
                        file_path = proc.info['exe']
                        features = []
                        if file_path and any(file_path.lower().endswith(ext) for ext in PE_EXTENSIONS):
                            pe_features = extract_features_pe(file_path)
                            if pe_features is not None:
                                pe_features = np.array(pe_features).reshape(1, -1)
                                y_pred_lgb = model_lgb.predict(pe_features)
                                dtest_xgb = xgb.DMatrix(pe_features)
                                y_pred_xgb = model_xgb.predict(dtest_xgb)
                                features = [y_pred_lgb[0], y_pred_xgb[0], 0.0] + behavior_features[:2]
                            else:
                                features = [0.0, 0.0, 0.0] + behavior_features[:2]
                        else:
                            features = [0.0, 0.0, 0.0] + behavior_features[:2]
                        
                        if features:
                            logging.info(f"Transformer input features (process {proc.info['name']}): {features}")
                            transformer_input = np.array(features).reshape(1, 1, -1)
                            y_pred_transformer = transformer_model.predict(transformer_input, verbose=0)
                            if y_pred_transformer[0] > 0.5:
                                logging.warning(f"⚠️ Suspicious process (transformer): {proc.info['name']}")
                                proc.terminate()
                            else:
                                logging.info(f"✅ Process is safe (transformer): {proc.info['name']}")
                    
                    net_connections = proc.net_connections() if hasattr(proc, 'net_connections') else []
                    for conn in net_connections:
                        if conn.raddr and any(pool in conn.raddr.ip for pool in MINING_POOLS):
                            logging.warning(f"⚠️ Suspicious network connection: {proc.info['name']} -> {conn.raddr.ip}")
                            proc.terminate()
                except Exception as e:
                    logging.error(f"Error analyzing process {proc.info['name']}: {e}")
        except Exception as e:
            logging.error(f"Monitoring error: {e}")
        time.sleep(10)