import gradio as gr
import tensorflow as tf
import numpy as np
import joblib as jb
from PIL import Image

# Load models
model = tf.keras.models.load_model("cleave_model_best_6_6.keras")
mlp_model = tf.keras.models.load_model("best_mlp.keras")

# Load scalers
scaler = jb.load("minmax_scaler.pkl")
tension_scaler = jb.load("tension_scaler.pkl")

# Preprocess image
def preprocess_image(image):
    img = image.resize((224, 224)).convert("RGB")
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    return img_array

# Predict cleave quality and optionally tension
def analyze_cleave(image, tension, angle):
    try:
        img_tensor = preprocess_image(image)
        features = scaler.transform([[angle, tension]])

        # CNN classification
        quality_pred = model.predict([img_tensor, features])[0][0]
        quality_label = "Good Cleave" if quality_pred > 0.5 else "Bad Cleave"

        if quality_pred < 0.5:
            # MLP regression
            angle_tensor = tf.convert_to_tensor(np.array([[angle]]), dtype=tf.float32)
            predicted_tension = mlp_model.predict([img_tensor, angle_tensor])
            optimal_tension = tension_scaler.inverse_transform(predicted_tension)[0][0]
            return quality_label, round(optimal_tension, 0)

        return quality_label, None

    except Exception as e:
        return f"Error: {str(e)}", None

# Gradio interface
inputs = [
    gr.Image(type="pil", label="Upload Cleave Image"),
    gr.Slider(400, 750, step=1, label="Cleave Tension (g)"),
    gr.Slider(0.0, 6.0, step=0.01, label="Cleave Angle (deg)")
]

outputs = [
    gr.Textbox(label="Cleave Prediction"),
    gr.Textbox(label="Suggested Optimal Tension (if bad cleave)")
]

gr.Interface(
    fn=analyze_cleave,
    inputs=inputs,
    outputs=outputs,
    title="Thorlabs Cleave Analyzer",
    description="Upload a cleave image and enter the tension and angle. The model will predict cleave quality and suggest optimal tension if bad."
).launch()
