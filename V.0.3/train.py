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
from tensorflow.keras import Model, Input
import logging
from utils import preprocess_labels, TransformerLayer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def train_models():
    EMBER_PATH = "C:/Users/Vovaaaan/Downloads/ember_dataset_2018_2/ember2018"
    X_TRAIN_PATH = os.path.join(EMBER_PATH, "X_train.dat")
    X_TEST_PATH = os.path.join(EMBER_PATH, "X_test.dat")

    if not (os.path.exists(X_TRAIN_PATH) and os.path.exists(X_TEST_PATH)):
        logging.info("🔧 Generating features...")
        ember.create_vectorized_features(EMBER_PATH)
    else:
        logging.info("✅ Features already created, skipping creation.")

    logging.info("📦 Loading training data...")
    X_train, y_train, X_test, y_test = ember.read_vectorized_features(EMBER_PATH)

    logging.info("🔄 Preprocessing labels...")
    y_train = preprocess_labels(y_train)
    y_test = preprocess_labels(y_test)

    logging.info("🧠 Training LightGBM...")
    dtrain = lgb.Dataset(X_train, label=y_train)
    params_lgb = {"objective": "binary", "metric": "auc"}
    model_lgb = lgb.train(params_lgb, dtrain)

    y_pred_lgb = model_lgb.predict(X_test)
    y_pred_lgb_binary = (y_pred_lgb > 0.5).astype(int)
    logging.info("📊 LightGBM Classification Report:")
    logging.info("\n%s", classification_report(y_test, y_pred_lgb_binary))
    logging.info("LightGBM ROC-AUC: %s", roc_auc_score(y_test, y_pred_lgb))

    logging.info("🧠 Training XGBoost...")
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

    y_pred_xgb = model_xgb.predict(dtest_xgb)
    y_pred_xgb_binary = (y_pred_xgb > 0.5).astype(int)
    logging.info("📊 XGBoost Classification Report:")
    logging.info("\n%s", classification_report(y_test, y_pred_xgb_binary))
    logging.info("XGBoost ROC-AUC: %s", roc_auc_score(y_test, y_pred_xgb))

    y_train_pred_lgb = model_lgb.predict(X_train)
    y_train_pred_xgb = model_xgb.predict(dtrain_xgb)
    y_test_pred_lgb = model_lgb.predict(X_test)
    y_test_pred_xgb = model_xgb.predict(dtest_xgb)

    X_meta_train = np.column_stack((y_train_pred_lgb, y_train_pred_xgb))
    y_meta_train = y_train
    X_meta_test = np.column_stack((y_test_pred_lgb, y_test_pred_xgb))
    y_meta_test = y_test

    logging.info("🧠 Training meta-model (neural network)...")
    meta_model = Sequential([
        Dense(10, activation='relu', input_shape=(X_meta_train.shape[1],)),
        Dense(1, activation='sigmoid')
    ])
    meta_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    meta_model.fit(X_meta_train, y_meta_train, epochs=10, batch_size=32, verbose=0)

    y_pred_meta = meta_model.predict(X_meta_test, verbose=0)
    y_pred_meta_binary = (y_pred_meta > 0.5).astype(int)
    logging.info("📊 Meta-model Classification Report:")
    logging.info("\n%s", classification_report(y_meta_test, y_pred_meta_binary))
    logging.info("Meta-model ROC-AUC: %s", roc_auc_score(y_meta_test, y_pred_meta))

    logging.info("🧠 Training transformer...")
    d_model = 64
    num_features = X_meta_train.shape[1] + 3  
    inputs = Input(shape=(1, num_features))
    projection = Dense(d_model)(inputs)
    x = TransformerLayer(num_heads=2, d_model=d_model, dff=128)(projection)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(10, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    transformer_model = Model(inputs=inputs, outputs=outputs)
    transformer_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    X_transformer_train = np.column_stack((X_meta_train, np.zeros((X_meta_train.shape[0], 3))))
    X_transformer_train = X_transformer_train.reshape((X_transformer_train.shape[0], 1, num_features))
    X_transformer_test = np.column_stack((X_meta_test, np.zeros((X_meta_test.shape[0], 3))))
    X_transformer_test = X_transformer_test.reshape((X_transformer_test.shape[0], 1, num_features))
    transformer_model.fit(X_transformer_train, y_meta_train, epochs=10, batch_size=32, verbose=0)

    y_pred_transformer = transformer_model.predict(X_transformer_test, verbose=0)
    y_pred_transformer_binary = (y_pred_transformer > 0.5).astype(int)
    logging.info("📊 Transformer Classification Report:")
    logging.info("\n%s", classification_report(y_meta_test, y_pred_transformer_binary))
    logging.info("Transformer ROC-AUC: %s", roc_auc_score(y_meta_test, y_pred_transformer))

    joblib.dump(model_lgb, "ember_model_lgb.lgb")
    model_xgb.save_model("xgb_model.json")
    meta_model.save("meta_model.h5")
    transformer_model.save("transformer_model.h5")
    logging.info("✅ Models saved.")

if __name__ == "__main__":
    train_models()