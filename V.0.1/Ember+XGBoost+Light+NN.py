import os
import ember
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import psutil
import time
import lief
import logging  


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  
    ]
)
def analyze_python_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().lower()
        suspicious_keywords = ["mining", "stratum", "pyminer", "cryptocurrency"]
        if any(keyword in content for keyword in suspicious_keywords):
            logging.warning(f"⚠️ Подозрительный Python файл: {file_path}")
            return True
        logging.info(f"✅ Python файл безопасен: {file_path}")
        return False
    except Exception as e:
        logging.error(f"Ошибка анализа Python файла {file_path}: {e}")
        return False
def preprocess_labels(y):
    """Преобразует метки: -1 -> 0, оставляет 0 и 1 без изменений."""
    y = np.where(y == -1, 0, y)  # Заменяем -1 на 0 (неклассифицированные -> безопасные)
    return y.astype(np.float32)  # Убедимся, что метки в правильном формате

def train_models():
    EMBER_PATH = "C:/Users/Vovaaaan/Downloads/ember_dataset_2018_2/ember2018"
    X_TRAIN_PATH = os.path.join(EMBER_PATH, "X_train.dat")
    X_TEST_PATH = os.path.join(EMBER_PATH, "X_test.dat")

    # === Проверка наличия сгенерированных признаков ===
    if not (os.path.exists(X_TRAIN_PATH) and os.path.exists(X_TEST_PATH)):
        logging.info("🔧 Генерация признаков...")
        ember.create_vectorized_features(EMBER_PATH)
    else:
        logging.info("✅ Признаки уже созданы, пропускаем создание.")

    # === Загрузка данных ===
    logging.info("📦 Загрузка обучающих данных...")
    X_train, y_train, X_test, y_test = ember.read_vectorized_features(EMBER_PATH)

    # === Предобработка меток ===
    logging.info("🔄 Предобработка меток...")
    y_train = preprocess_labels(y_train)
    y_test = preprocess_labels(y_test)

    # === Обучение LightGBM ===
    logging.info("🧠 Обучение LightGBM...")
    dtrain = lgb.Dataset(X_train, label=y_train)
    params_lgb = {"objective": "binary", "metric": "auc"}
    model_lgb = lgb.train(params_lgb, dtrain)

    # === Предсказания LightGBM ===
    y_pred_lgb = model_lgb.predict(X_test)
    y_pred_lgb_binary = (y_pred_lgb > 0.5).astype(int)
    logging.info("📊 LightGBM Classification Report:")
    logging.info("\n%s", classification_report(y_test, y_pred_lgb_binary))
    logging.info("LightGBM ROC-AUC: %s", roc_auc_score(y_test, y_pred_lgb))

    # === Обучение XGBoost ===
    logging.info("🧠 Обучение XGBoost...")
    dtrain_xgb = xgb.DMatrix(X_train, label=y_train)
    dtest_xgb = xgb.DMatrix(X_test, label=y_test)
    params_xgb = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'eta': 0.1,
        'seed': 42
    }
    model_xgb = xgb.train(params_xgb, dtrain_xgb, num_boost_round=100)

    # === Предсказания XGBoost ===
    y_pred_xgb = model_xgb.predict(dtest_xgb)
    y_pred_xgb_binary = (y_pred_xgb > 0.5).astype(int)
    logging.info("📊 XGBoost Classification Report:")
    logging.info("\n%s", classification_report(y_test, y_pred_xgb_binary))
    logging.info("XGBoost ROC-AUC: %s", roc_auc_score(y_test, y_pred_xgb))

    # === Получение предсказаний для мета-модели ===
    y_train_pred_lgb = model_lgb.predict(X_train)
    y_train_pred_xgb = model_xgb.predict(dtrain_xgb)
    y_test_pred_lgb = model_lgb.predict(X_test)
    y_test_pred_xgb = model_xgb.predict(dtest_xgb)

    # === Создание наборов для мета-модели ===
    X_meta_train = np.column_stack((y_train_pred_lgb, y_train_pred_xgb))
    y_meta_train = y_train
    X_meta_test = np.column_stack((y_test_pred_lgb, y_test_pred_xgb))
    y_meta_test = y_test

    # === Обучение мета-модели (нейросети) ===
    logging.info("🧠 Обучение мета-модели (нейросети)...")
    meta_model = Sequential([
        Dense(10, activation='relu', input_shape=(X_meta_train.shape[1],)),
        Dense(1, activation='sigmoid')
    ])
    meta_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    meta_model.fit(X_meta_train, y_meta_train, epochs=10, batch_size=32, verbose=0)

    # === Предсказания мета-модели ===
    y_pred_meta = meta_model.predict(X_meta_test, verbose=0)
    y_pred_meta_binary = (y_pred_meta > 0.5).astype(int)
    logging.info("📊 Meta-model Classification Report:")
    logging.info("\n%s", classification_report(y_meta_test, y_pred_meta_binary))
    logging.info("Meta-model ROC-AUC: %s", roc_auc_score(y_meta_test, y_pred_meta))

    # === Сохранение моделей ===
    joblib.dump(model_lgb, "ember_model_lgb.lgb")
    model_xgb.save_model("xgb_model.json")
    meta_model.save("meta_model.h5")
    logging.info("✅ Модели сохранены.")

def extract_features(file_path):
    """Извлекает признаки из PE-файла с помощью ember."""
    try:
        logging.info(f"Проверка файла: {file_path}")
        # Проверяем, является ли файл валидным PE-файлом
        if not lief.is_pe(file_path):
            logging.warning(f"{file_path} не является валидным PE-файлом")
            return None
        with open(file_path, "rb") as f:
            binary = f.read()
        extractor = ember.PEFeatureExtractor()
        features = np.array(extractor.feature_vector(binary), dtype=np.float32)
        logging.info(f"Признаки успешно извлечены из {file_path}")
        return features
    except Exception as e:
        logging.error(f"Ошибка извлечения признаков из {file_path}: {e}")
        return None

def monitor_system():
    logging.info("🚀 Запуск мониторинга...")
    # Загрузка моделей
    try:
        model_lgb = joblib.load("ember_model_lgb.lgb")
        model_xgb = xgb.Booster()
        model_xgb.load_model("xgb_model.json")
        model_meta = tf.keras.models.load_model("meta_model.h5")
        logging.info("✅ Модели успешно загружены.")
    except Exception as e:
        logging.error(f"Ошибка загрузки моделей: {e}")
        return

    MONITOR_DIR = "C:/Users/Vovaaaan/Desktop/lessons_space"  # Директория для мониторинга
    checked_files = set()  # Чтобы не проверять файлы повторно
    MINING_POOLS = ["pool.minergate.com", "xmr.pool.minergate.com", "pool.nicehash.com"]  # Список майнинговых пулов

    while True:
        try:
                        # Мониторинг новых файлов
            exe_files = [f for f in os.listdir(MONITOR_DIR) if f.endswith(".exe") and f not in checked_files]
            py_files = [f for f in os.listdir(MONITOR_DIR) if f.endswith(".py") and f not in checked_files]
            if not exe_files and not py_files:
                logging.info(f"В директории {MONITOR_DIR} нет новых .exe или .py файлов для проверки")
            for file in exe_files:
                file_path = os.path.join(MONITOR_DIR, file)
                features = extract_features(file_path)
                if features is not None:
                    features = np.array(features).reshape(1, -1)
                    y_pred_lgb = model_lgb.predict(features)
                    dtest_xgb = xgb.DMatrix(features)
                    y_pred_xgb = model_xgb.predict(dtest_xgb)
                    X_meta = np.column_stack((y_pred_lgb, y_pred_xgb))
                    y_pred_meta = model_meta.predict(X_meta, verbose=0)
                    if y_pred_meta[0] > 0.5:
                        logging.warning(f"⚠️ Обнаружен вредоносный файл: {file_path}")
                    else:
                        logging.info(f"✅ Файл безопасен: {file_path}")
                checked_files.add(file)
            for file in py_files:
                file_path = os.path.join(MONITOR_DIR, file)
                if analyze_python_file(file_path):
                    logging.warning(f"⚠️ Обнаружен подозрительный Python файл: {file_path}")
                else:
                    logging.info(f"✅ Python файл безопасен: {file_path}")
                checked_files.add(file)

            # Мониторинг процессов (для обнаружения майнинга)
            for proc in psutil.process_iter(['name', 'exe', 'cpu_percent']):
                try:
                    if proc.info['cpu_percent'] > 80:  # Подозрительно высокая нагрузка
                        file_path = proc.info['exe']
                        if file_path and file_path.endswith(".exe"):
                            features = extract_features(file_path)
                            if features is not None:
                                features = np.array(features).reshape(1, -1)
                                y_pred_lgb = model_lgb.predict(features)
                                dtest_xgb = xgb.DMatrix(features)
                                y_pred_xgb = model_xgb.predict(dtest_xgb)
                                X_meta = np.column_stack((y_pred_lgb, y_pred_xgb))
                                y_pred_meta = model_meta.predict(X_meta, verbose=0)
                                if y_pred_meta[0] > 0.5:
                                    logging.warning(f"⚠️ Подозрительный процесс: {proc.info['name']}")
                                    proc.terminate()
                                    # Здесь можно добавить оповещение (например, email)
                    # Проверка сетевой активности (для майнинга)
                    net_connections = proc.net_connections() if hasattr(proc, 'net_connections') else []
                    for conn in net_connections:
                        if conn.raddr and any(pool in conn.raddr.ip for pool in MINING_POOLS):
                            logging.warning(f"⚠️ Подозрительное сетевое подключение: {proc.info['name']} -> {conn.raddr.ip}")
                            proc.terminate()
                except Exception as e:
                    logging.error(f"Ошибка при анализе процесса {proc.info['name']}: {e}")
        except Exception as e:
            logging.error(f"Ошибка мониторинга: {e}")
        time.sleep(10)  # Проверяем каждые 10 секунд

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    # train_models()  # Раскомментируй для обучения
    monitor_system()
