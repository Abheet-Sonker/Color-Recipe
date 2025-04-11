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

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

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
    # RED: Medium to dark red tones
    if 30 <= L <= 70 and 40 <= a <= 80 and -10 <= b <= 30:
        return joblib.load("Red_Family.pkl"), joblib.load("scaler_Red.pkl"), "Red"
    
    # ORANGE: Bright warm tones
    elif 40 <= L <= 85 and 15 <= a <= 45 and 30 <= b <= 70:
        return joblib.load("Orange_Family.pkl"), joblib.load("scaler_Orange.pkl"), "Orange"
    
    # GREEN: All tones of green
    elif 30 <= L <= 85 and -80 <= a <= -10 and -20 <= b <= 30:
        return joblib.load("Green_Family.pkl"), joblib.load("scaler_Green.pkl"), "Green"
    
    # BLUE: Deep to lighter blues
    elif 20 <= L <= 70 and -20 <= a <= 20 and -80 <= b <= -10:
        return joblib.load("Blue_Family.pkl"), joblib.load("scaler_Blue.pkl"), "Blue"
    
    # YELLOW: Pale to deep yellows
    elif 60 <= L <= 100 and -15 <= a <= 30 and 40 <= b <= 80:
        return joblib.load("Yellow_Family.pkl"), joblib.load("scaler_Yellow.pkl"), "Yellow"
    
    # PURPLE: Dark violets to reddish-purples
    elif 20 <= L <= 60 and 20 <= a <= 60 and -50 <= b <= -5:
        return joblib.load("Purple_Family.pkl"), joblib.load("scaler_Purple.pkl"), "Purple"
    
    # BROWN: Dark oranges/yellows with lower lightness
    elif 20 <= L <= 55 and 10 <= a <= 30 and 10 <= b <= 40:
        return joblib.load("Brown_Family.pkl"), joblib.load("scaler_Brown.pkl"), "Brown"
    
    # PINK: Light reds, high L, positive a
    elif 65 <= L <= 100 and 20 <= a <= 50 and 0 <= b <= 30:
        return joblib.load("Pink_Family.pkl"), joblib.load("scaler_Pink.pkl"), "Pink"
    
    # GRAY: Low chroma (a, b near 0)
    elif 25 <= L <= 90 and -5 <= a <= 5 and -5 <= b <= 5:
        return joblib.load("Gray_Family.pkl"), joblib.load("scaler_Gray.pkl"), "Gray"
    
    # BLACK: Very low lightness, near neutral
    elif 0 <= L <= 25 and -10 <= a <= 10 and -10 <= b <= 10:
        return joblib.load("Black_Family.pkl"), joblib.load("scaler_Black.pkl"), "Black"
    
    # WHITE: Very high lightness, low chroma
    elif 90 <= L <= 100 and -5 <= a <= 5 and -5 <= b <= 5:
        return joblib.load("White_Family.pkl"), joblib.load("scaler_White.pkl"), "White"
    
    else:
        return None, None, "Unknown"


# Load dataset to get pigment names
@st.cache_data
def load_dataset(color_family):
    filename = f"{color_family}_family_dataset.csv"
    try:
        df = pd.read_csv(filename)
        pigment_columns = [col for col in df.columns if col.startswith("Pigment")]
        return pigment_columns
    except:
        return []

# Delta E (CIE76)
def delta_e_cie76(lab1, lab2):
    return np.linalg.norm(lab1 - lab2)

# Optimization function
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
    return result,target_lab

# Main app logic
if uploaded_file:
    pil_image = Image.open(uploaded_file).convert('RGB')
    st.image(pil_image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Calculate LAB and Match Pigments"):
        lab_color, L, a, b = extract_avg_lab(pil_image)
        st.markdown(f"**🎯 Average LAB Color:** `L* = {L:.2f}, a* = {a:.2f}, b* = {b:.2f}`")

        model, scaler, color_family = load_model(L, a, b)
        pigment_columns = load_dataset(color_family)

        if model is None or not pigment_columns:
            st.error("❌ Could not determine a matching color family or load required model/data.")
        else:
            st.success(f"🎨 Detected Color Family: **{color_family}**")
            result,target_lab = optimize_pigments(lab_color, model, scaler, pigment_columns)

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
                converted_predicted_lab = np.round(predicted_lab).astype(np.float32).reshape(1, 1, 3)
                converted_target_lab = np.round(target_lab).astype(np.float32).reshape(1, 1, 3)
                
                lab_color_target= converted_target_lab
                lab_color_predicted = converted_predicted_lab
                
                rgb_color1 = lab2rgb(lab_color_target)
                rgb_color2 = lab2rgb(lab_color_predicted)
        
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))

                ax[0].imshow(rgb_color1)
                ax[0].axis('off')
                ax[0].set_title('Target color')
                
                ax[1].imshow(rgb_color2)
                ax[1].axis('off')
                ax[1].set_title('Predicted color')

                st.pyplot(fig)
            else:
                st.error("Optimization failed to converge. Try another image or color.")
