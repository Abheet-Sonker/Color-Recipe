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

# --- Excel Export Helper ---
@st.cache_data
def convert_df_to_excel(df):
    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Pigment Recipes')
    return output.getvalue()

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Color Recipe Optimizer", layout="wide")
st.title("🎨 Color Recipe Optimizer (Multi-Color with Excel Export)")

# --- Load model based on LAB range ---
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
    try:
        df = pd.read_csv(f"{color_family}_family_dataset.csv")
        return [col for col in df.columns if col.startswith("Pigment")]
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
    return result

# --- Input Section ---
st.markdown("### 📁 Optional Excel Input")
input_method = st.radio("Choose input method:", 
                       ["Upload Excel File", "Manual Input"], 
                       index=1)

color_data = []

if input_method == "Upload Excel File":
    uploaded_file = st.file_uploader("Upload Excel file (columns: Name, L, a, b, Qty_kg)", type="xlsx")
    
    if uploaded_file:
        try:
            df_upload = pd.read_excel(uploaded_file)
            required_cols = ['Name', 'L', 'a', 'b', 'Qty_kg']
            
            if all(col in df_upload.columns for col in required_cols):
                color_data = df_upload[required_cols].to_dict('records')
                st.success(f"✅ Successfully loaded {len(color_data)} colors from file!")
                
                with st.expander("Preview uploaded data"):
                    st.dataframe(df_upload[required_cols])
            else:
                st.error("❌ Missing required columns. Needed columns: Name, L, a, b, Qty_kg")
                
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
else:
    # --- Manual Input Section ---
    st.markdown("### 🧾 Input Multiple Target LAB Colors and Quantities")
    num_rows = st.number_input("How many target colors do you want to enter?", min_value=1, max_value=20, value=1)
    
    for i in range(int(num_rows)):
        st.markdown(f"#### 🎯 Target Color {i+1}")
        name = st.text_input(f"Color Name", key=f"name_{i}")
        qty = st.number_input(f"Quantity to produce (kg)", min_value=0.1, value=1.0, key=f"qty_{i}")
        L = st.number_input(f"L*", min_value=0.0, max_value=100.0, step=0.1, key=f"L_{i}")
        a = st.number_input(f"a*", min_value=-128.0, max_value=127.0, step=0.1, key=f"a_{i}")
        b = st.number_input(f"b*", min_value=-128.0, max_value=127.0, step=0.1, key=f"b_{i}")
        color_data.append({'Name': name, 'L': L, 'a': a, 'b': b, 'Qty_kg': qty})

# --- Optimization Section ---
results = []

if st.button("🚀 Generate Recipes and Export Excel"):
    for color in color_data:
        L, a, b = color['L'], color['a'], color['b']
        lab = np.array([[L, a, b]], dtype=np.float32)

        model, scaler, color_family = load_model(L, a, b)
        pigment_columns = load_dataset(color_family)

        if model is None or not pigment_columns:
            st.warning(f"⚠️ Skipped '{color['Name']}' – Unknown color family.")
            continue

        result = optimize_pigments(lab, model, scaler, pigment_columns)

        if result.success:
            optimized_pct = result.x
            pigment_kg = optimized_pct * (color['Qty_kg'] / 100)

            predicted_lab_scaled = model.predict([optimized_pct])[0]
            predicted_lab = scaler.inverse_transform(predicted_lab_scaled.reshape(1, -1))[0]
            delta_e = delta_e_cie76(lab.flatten(), predicted_lab)
            rmse = np.sqrt(mean_squared_error(lab, predicted_lab.reshape(1, -1), multioutput='raw_values'))

            row = {
                'Name': color['Name'],
                'L*': L, 'a*': a, 'b*': b,
                'Qty (kg)': color['Qty_kg'],
                'Color Family': color_family,
                'Pred L*': round(predicted_lab[0], 2),
                'Pred a*': round(predicted_lab[1], 2),
                'Pred b*': round(predicted_lab[2], 2),
                'ΔE': round(delta_e, 2),
                'RMSE_L': round(rmse[0], 2),
                'RMSE_a': round(rmse[1], 2),
                'RMSE_b': round(rmse[2], 2)
            }

            for p, qty_kg in zip(pigment_columns, pigment_kg):
                if '(%)' in p:
                    kg_column = p.replace('(%)', '(kg)')
                else:
                    kg_column = p + ' (kg)'
                row[kg_column] = round(qty_kg, 3)

            results.append(row)

            # Visualize LAB Target vs Prediction
            target_lab_patch = np.round(lab).astype(np.float32).reshape(1, 1, 3)
            predicted_lab_patch = np.round(predicted_lab).astype(np.float32).reshape(1, 1, 3)

            target_rgb = lab2rgb(target_lab_patch)
            predicted_rgb = lab2rgb(predicted_lab_patch)

            fig, ax = plt.subplots(1, 2, figsize=(6, 1))
            ax[0].imshow(target_rgb)
            ax[0].set_title('Target')
            ax[0].axis('off')
            
            ax[1].imshow(predicted_rgb)
            ax[1].set_title('Predicted')
            ax[1].axis('off')
            
            plt.tight_layout(pad=0.01)
            st.markdown(f"##### 🔍 {color['Name']} – ΔE = `{delta_e:.2f}`")
            st.pyplot(fig)

        else:
            st.warning(f"❌ Optimization failed for '{color['Name']}'.")

    if results:
        df_results = pd.DataFrame(results)
        st.dataframe(df_results)
        st.success("✅ All recipes generated successfully!")
        excel_data = convert_df_to_excel(df_results)
        st.download_button("📥 Download Excel", data=excel_data, 
                          file_name="Pigment_Recipes.xlsx", 
                          mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
