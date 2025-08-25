

import os
import ember
import joblib
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score
def main():
    EMBER_PATH =  "ember_dataset_2018_2/ember2018"
    X_TRAIN_PATH = os.path.join(EMBER_PATH, "X_train.dat")
    X_TEST_PATH = os.path.join(EMBER_PATH, "X_test.dat")


    if not (os.path.exists(X_TRAIN_PATH) and os.path.exists(X_TEST_PATH)):
            ember.create_vectorized_features(EMBER_PATH)
		print(“DONE”)
    else:
        print("✅ The signs have already been created, skip the creation.")


    print("📦 Loading training data...")
    X_train, y_train, X_test, y_test = ember.read_vectorized_features(EMBER_PATH)


    dtrain = lgb.Dataset(X_train, label=y_train)
    params = {"objective": "binary", "metric": "auc"}
    model = lgb.train(params, dtrain)


    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)


    print(classification_report(y_test, y_pred_binary))
    print("ROC-AUC:", roc_auc_score(y_test, y_pred))    
    joblib.dump(model, "ember_model.lgb")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()

