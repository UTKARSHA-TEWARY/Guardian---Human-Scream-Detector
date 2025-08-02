import numpy as np
import joblib
from sklearn.svm import SVC

# Dummy X and y if retraining
# X = ...
# y = ...
# model = SVC()
# model.fit(X, y)

# Load your already-trained model here (if needed)
model = joblib.load("svm_model.pkl")

# Force model to use clean numpy types
model.support_ = np.array(model.support_, dtype=np.int32)
model.support_vectors_ = np.array(model.support_vectors_, dtype=np.float32)

joblib.dump(model, 'svm_model_clean.pkl')
print("✅ Saved cleaned model as svm_model_clean.pkl")
