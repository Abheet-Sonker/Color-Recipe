import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import joblib
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from skimage.color import lab2rgb
import matplotlib.pyplot as plt

st.set_page_config(page_title="Color Recipe Optimizer", layout="centered")
st.title("🎨 Color Recipe Optimizer (Using Trained Models)")

# Upload image or manual LAB
st.markdown("### 🖼️ Input Options (Upload Image or Enter LAB Values)")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

manual_L = st.number_input("L*", min_value=0.0, max_value=100.0, step=0.1, format="%.2f")
manual_a = st.number_input("a*", min_value=-128.0, max_value=127.0, step=0.1, format="%.2f")
manual_b = st.number_input("b*", min_value=-128.0, max_value=127.0, step=0.1, format="%.2f")

# Helper function to convert image to LAB
def extract_avg_lab(pil_image):
    img = np.array(pil_image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    lab_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    avg_lab = np.mean(lab_img.reshape(-1, 3), axis=0)
    L = (avg_lab[0] / 255) * 100
    a = avg_lab[1] - 128
    b = avg_lab[2] - 128
    return np.array([[L, a, b]], dtype=np.float32), L, a, b

# Load model and scaler based on LAB
@st.cache_resource
def load_model(L, a, b):
    if 30 <= L <= 70 and 40 <= a <= 100 and -10 <= b <= 40:
        return joblib.load("Red_Family.pkl"), joblib.load("Scaler_Red.pkl"), "Red"
    elif 50 <= L <= 85 and 20 <= a <= 60 and 30 <= b <= 80:
        return joblib.load("Orange_Family.pkl"), joblib.load("Scaler_Orange.pkl"), "Orange"
    elif 30 <= L <= 80 and -80 <= a <= -10 and -30 <= b <= 30:
        return joblib.load("Green_Family.pkl"), joblib.load("Scaler_Green.pkl"), "Green"
    elif 20 <= L <= 70 and -20 <= a <= 30 and -100 <= b <= -10:
        return joblib.load("Blue_Family.pkl"), joblib.load("Scaler_Blue.pkl"), "Blue"
    elif 60 <= L <= 100 and -12 <= a <= 30 and 40 <= b <= 100:
        return joblib.load("Yellow_Family.pkl"), joblib.load("Scaler_Yellow.pkl"), "Yellow"
    elif 20 <= L <= 80 and 20 <= a <= 60 and -60 <= b <= -5:
        return joblib.load("Purple_Family.pkl"), joblib.load("Scaler_Purple.pkl"), "Purple"
    elif 20 <= L <= 60 and 10 <= a <= 40 and 10 <= b <= 50:
        return joblib.load("Brown_Family.pkl"), joblib.load("Scaler_Brown.pkl"), "Brown"
    elif 20 <= L <= 100 and 20 <= a <= 10 and -10 <= b <= 10:
        return joblib.load("Gray_Family.pkl"), joblib.load("Scaler_Gray.pkl"), "Gray"
    elif 0 <= L <= 20 and -5 <= a <= 5 and -5 <= b <= 5:
        return joblib.load("Black_Family.pkl"), joblib.load("Scaler_Black.pkl"), "Black"
    elif 90 <= L <= 100 and -5 <= a <= 5 and -5 <= b <= 5:
        return joblib.load("White_Family.pkl"), joblib.load("Scaler_White.pkl"), "White"
    else:
        return None, None, "Unknown"

@st.cache_data
def load_dataset(color_family):
    filename = f"{color_family}_family_dataset.csv"
    try:
        df = pd.read_csv(filename)
        pigment_columns = [col for col in df.columns if col.startswith("Pigment")]
        return pigment_columns
    except:
        return []

def delta_e_cie76(lab1, lab2):
    return np.linalg.norm(lab1 - lab2)

def optimize_pigments(target_lab, model, scaler, pigment_columns):
    def loss_function(pigments):
        pigments = np.clip(pigments, 0, 100)
        predicted_lab_scaled = model.predict([pigments])[0]
        predicted_lab = scaler.inverse_transform(predicted_lab_scaled.reshape(1, -1))[0]
        return delta_e_cie76(target_lab.flatten(), predicted_lab)

    initial_guess = np.full(len(pigment_columns), 100 / len(pigment_columns))
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 100}
    bounds = [(0, 100) for _ in pigment_columns]

    result = minimize(loss_function, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result, target_lab

# Main app logic
if st.button("🔍 Calculate LAB and Match Pigments"):
    if uploaded_file:
        pil_image = Image.open(uploaded_file).convert('RGB')
        st.image(pil_image, caption="Uploaded Image", use_column_width=True)
        lab_color, L, a, b = extract_avg_lab(pil_image)
        st.markdown(f"**🎯 Average LAB Color from Image:** `L* = {L:.2f}, a* = {a:.2f}, b* = {b:.2f}`")
    elif any([manual_L, manual_a, manual_b]):
        lab_color = np.array([[manual_L, manual_a, manual_b]], dtype=np.float32)
        L, a, b = manual_L, manual_a, manual_b
        st.markdown(f"**🎯 Manual LAB Input:** `L* = {L:.2f}, a* = {a:.2f}, b* = {b:.2f}`")
    else:
        st.error("❗ Please upload an image or enter LAB values manually.")
        st.stop()

    # Visualize input LAB color
    input_lab_patch = np.round(lab_color).astype(np.float32).reshape(1, 1, 3)
    input_rgb = lab2rgb(input_lab_patch)
    fig_input, ax_input = plt.subplots(figsize=(2, 2))
    ax_input.imshow(input_rgb)
    ax_input.axis('off')
    ax_input.set_title("Input LAB Color")
    st.pyplot(fig_input)

    model, scaler, color_family = load_model(L, a, b)
    pigment_columns = load_dataset(color_family)

    if model is None or not pigment_columns:
        st.error("❌ Could not determine a matching color family or load required model/data.")
    else:
        st.success(f"🎨 Detected Color Family: **{color_family}**")
        result, target_lab = optimize_pigments(lab_color, model, scaler, pigment_columns)

        if result.success:
            optimal_pigments = result.x
            predicted_lab_scaled = model.predict([optimal_pigments])[0]
            predicted_lab = scaler.inverse_transform(predicted_lab_scaled.reshape(1, -1))[0]
            delta_e = delta_e_cie76(lab_color.flatten(), predicted_lab)
            rmse = np.sqrt(mean_squared_error(lab_color, predicted_lab.reshape(1, -1), multioutput='raw_values'))

            st.subheader("🧪 Optimal Pigment Recipe:")
            for name, pct in zip(pigment_columns, optimal_pigments):
                st.markdown(f"- **{name}**: {pct:.2f}%")

            st.subheader("📊 Prediction & Accuracy:")
            st.markdown(f"**Predicted LAB**: `L* = {predicted_lab[0]:.2f}, a* = {predicted_lab[1]:.2f}, b* = {predicted_lab[2]:.2f}`")
            st.markdown(f"**ΔE (CIE76)**: `{delta_e:.2f}`")
            st.markdown(f"**RMSE**: `L* = {rmse[0]:.2f}, a* = {rmse[1]:.2f}, b* = {rmse[2]:.2f}`")

            # Visualization of predicted vs target
            converted_predicted_lab = np.round(predicted_lab).astype(np.float32).reshape(1, 1, 3)
            converted_target_lab = np.round(target_lab).astype(np.float32).reshape(1, 1, 3)

            rgb_color1 = lab2rgb(converted_target_lab)
            rgb_color2 = lab2rgb(converted_predicted_lab)

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(rgb_color1)
            ax[0].axis('off')
            ax[0].set_title('Target Color')

            ax[1].imshow(rgb_color2)
            ax[1].axis('off')
            ax[1].set_title('Predicted Color')

            st.pyplot(fig)
        else:
            st.error("❌ Optimization failed to converge. Try another image or LAB values.")
