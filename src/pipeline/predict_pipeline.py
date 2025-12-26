from pathlib import Path
import pickle
import pandas as pd


class PredictPipeline:

    def __init__(self):
        # Start from this file and walk UP until we find artifacts/
        current_file = Path(__file__).resolve()

        project_root = None
        for parent in current_file.parents:
            if (parent / "artifacts").is_dir():
                project_root = parent
                break

        if project_root is None:
            raise FileNotFoundError(
                "Could not locate project root. 'artifacts/' directory not found."
            )

        preprocessor_path = project_root / "artifacts" / "preprocessor.pkl"
        model_path = project_root / "artifacts" / "model.pkl"

        print("✅ Project root found at:", project_root)
        print("✅ Preprocessor path:", preprocessor_path)
        print("✅ Model path:", model_path)

        with open(preprocessor_path, "rb") as f:
            self.preprocessor = pickle.load(f)

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, input_data: dict):
        df = pd.DataFrame([input_data])
        transformed_data = self.preprocessor.transform(df)
        return self.model.predict(transformed_data)[0]
