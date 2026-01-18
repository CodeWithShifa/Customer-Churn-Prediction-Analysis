import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


try:
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print("✅ Data Load Ho Gaya!")
except:
    print("❌ Error: CSV file nahi mili! Check karein ke file isi folder mein hai.")

# 2. Cleaning & Preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)

# Text ko numbers mein badalna
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# 3. Features aur Target set karna
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling (Deep Learning ke liye must hai)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. AAP KA DEEP LEARNING MODEL
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)), 
    Dropout(0.2), 
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid') 
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Training Shuru (Epochs)
print("\n--- Training Start (30 Rounds) ---")
model.fit(X_train, y_train, epochs=30, batch_size=10, verbose=1)

# 6. Feature Importance (Simple Weights Analysis)
weights = model.layers[0].get_weights()[0]
importance = np.mean(np.abs(weights), axis=1)
feature_names = df.drop('Churn', axis=1).columns

# Graph dikhana
plt.figure(figsize=(10,6))
plt.barh(feature_names, importance, color='skyblue')
plt.xlabel("Importance Score")
plt.title("Kaunse Factors Churn par asar kar rahe hain?")
plt.show()
model.save('my_churn_model.keras')
print("✅ Model Save Ho Gaya!")