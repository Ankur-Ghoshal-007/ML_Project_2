import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score


class ModelTrainer:

    def __init__(self):
        self.models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        }

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):

        best_model = None
        best_recall = 0.0
        best_model_name = None

        for model_name, model in self.models.items():

            print(f"\nTraining {model_name}...")

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            recall = recall_score(y_test, y_pred)

            print("Classification Report:")
            print(classification_report(y_test, y_pred))

            if recall > best_recall:
                best_recall = recall
                best_model = model
                best_model_name = model_name

        print(f"\nBest Model: {best_model_name}")
        print(f"Best Recall: {best_recall}")

        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/model.pkl", "wb") as f:
            pickle.dump(best_model, f)

        return best_model_name, best_recall
