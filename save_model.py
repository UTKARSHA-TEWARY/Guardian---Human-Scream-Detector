import numpy as np
import joblib
from sklearn.svm import SVC

# Generate dummy training data
X = np.random.rand(100, 60)  # 100 samples, 60 MFCC features
y = np.random.randint(0, 2, 100)  # Binary labels

# Train SVM model
model = SVC(probability=True)
model.fit(X, y)

# Save clean model
joblib.dump(model, 'svm_model_clean.pkl')
print("✅ Saved freshly trained model as svm_model_clean.pkl")
