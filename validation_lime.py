import sys
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer

def test_lime_functional():
    print("--- Starting LIME Functional Verification ---")
    
    try:
        # 1. Setup minimal data
        iris = load_iris()
        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        rf.fit(iris.data, iris.target)
        
        # 2. Initialize Explainer
        # This checks if the sklearn/numpy integration is intact
        print("--> Initializing Explainer...")
        explainer = LimeTabularExplainer(
            iris.data, 
            feature_names=iris.feature_names,
            class_names=iris.target_names,
            discretize_continuous=True
        )

        # 3. TRIGGER THE DRIFT
        # Generating an explanation triggers discretization logic.
        # NumPy 2.x will fail here if legacy aliases (np.float, np.int) are used internally.
        print("--> Generating Explanation for test instance...")
        exp = explainer.explain_instance(iris.data[0], rf.predict_proba, num_features=2)
        
        if exp is not None:
            print("    [✓] Explanation successfully generated.")
        
        print("--- SMOKE TEST PASSED ---")

    except AttributeError as ae:
        print(f"CRITICAL VALIDATION FAILURE: {str(ae)}")
        # If NumPy 2.x is present, expect: "module 'numpy' has no attribute 'float'"
        sys.exit(1)
    except Exception as e:
        print(f"UNEXPECTED FAILURE: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_lime_functional()