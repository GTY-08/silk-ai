import os, json, colorsys
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import joblib

DATA_PATH = "data/SILK_emotion_template.csv"
ARTIFACTS_DIR = "artifacts"
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
need = {"color_hex","shape","sound","label"}
miss = need - set(df.columns)
if miss:
    raise ValueError(f"missing columns: {miss}")

def hex_to_hsv(x: str):
    x = str(x).strip().lstrip("#")
    if len(x) == 3:
        x = "".join(c*2 for c in x)
    r = int(x[0:2],16)/255.0
    g = int(x[2:4],16)/255.0
    b = int(x[4:6],16)/255.0
    h,s,v = colorsys.rgb_to_hsv(r,g,b)
    return [h,s,v]

df[["h","s","v"]] = df["color_hex"].apply(lambda c: pd.Series(hex_to_hsv(c)))

shape_enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
sound_enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
shape_enc.fit(df[["shape"]])
sound_enc.fit(df[["sound"]])
n_shape = len(shape_enc.categories_[0])
n_sound = len(sound_enc.categories_[0])

label_enc = LabelEncoder()
y = label_enc.fit_transform(df["label"].astype(str))

scaler = StandardScaler()
X_hsv = df[["h","s","v"]].values
X_hsv_scaled = scaler.fit_transform(X_hsv)

zeros_shape = np.zeros((len(df), n_shape), dtype=np.float32)
zeros_sound = np.zeros((len(df), n_sound), dtype=np.float32)
X = np.hstack([X_hsv_scaled, zeros_shape, zeros_sound])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(label_enc.classes_), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=1)

model.save(os.path.join(MODEL_DIR, "keras_model.h5"))
joblib.dump(shape_enc, os.path.join(ARTIFACTS_DIR, "shape_encoder.pkl"))
joblib.dump(sound_enc, os.path.join(ARTIFACTS_DIR, "sound_encoder.pkl"))
joblib.dump(label_enc, os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"))
joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.pkl"))

meta = {
    "shape_categories": shape_enc.categories_[0].tolist(),
    "sound_categories": sound_enc.categories_[0].tolist(),
    "class_names": label_enc.classes_.tolist(),
    "feature_order": ["h","s","v"] + [f"shape:{c}" for c in shape_enc.categories_[0]] + [f"sound:{c}" for c in sound_enc.categories_[0]],
    "note": "shape/sound inputs are zeroed; model learns color-only"
}
with open(os.path.join(ARTIFACTS_DIR, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("DONE")
