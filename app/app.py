import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Function to initialize the model
def get_classifier(bll=1053, fsrq=551, features=7):
    output_bias = tf.keras.initializers.Constant(np.log([fsrq / bll]))
    standard_nn_classifier = tf.keras.models.Sequential([
        tf.keras.layers.Dense(42, activation='relu', input_shape=(features,)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
    ])
    return standard_nn_classifier

st.title("BCU Classifier Web App")

# Load the model
standard_nn_classifier = get_classifier()

# Define the log transformations
def log_transform(value, base=0):
    return np.log(value + base)

def predict_class(pred_value):
    return "FSRQ" if round(pred_value) == 1 else "BLL"

# User input for 7 features
pl_index = st.number_input("PL_Index", value=2.116692,format="%.25f")
nu_syn = st.number_input("nu_syn", value=9.120108e+13,format="%.25f")
lp_index = st.number_input("LP_Index", value=2.078927,format="%.25f")
pivot_energy = st.number_input("Pivot_Energy", value=1721.603882,format="%.25f")
frac_variability = st.number_input("Frac_Variability", value=0.406565,format="%.25f")
variability_index = st.number_input("Variability_Index", value=46.780693,format="%.25f")
nuFnu_syn = st.number_input("nuFnu_syn", value=1.936770e-12, step=1e-15,format="%.25f")

# Apply transformations
feature_columns = ['PL_Index', 'nu_syn', 'LP_Index', 'Pivot_Energy', 'Frac_Variability', 'Variability_Index', 'nuFnu_syn']
transformer = make_column_transformer(
    (StandardScaler(), 
        feature_columns)
)
X_train = pd.read_csv('./classification_data//train_samples.csv')
transformer.fit(X_train)
pivot_energy = log_transform(pivot_energy)
variability_index = log_transform(variability_index)
nu_syn = log_transform(nu_syn, base=1e12)
nuFnu_syn = log_transform(nuFnu_syn, base=1e-12)

# Create input array for prediction
data = {
    'PL_Index': [pl_index],
    'nu_syn': [nu_syn],
    'LP_Index': [lp_index],
    'Pivot_Energy': [pivot_energy],
    'Frac_Variability': [frac_variability],
    'Variability_Index': [variability_index],
    'nuFnu_syn': [nuFnu_syn]
}

# Create a DataFrame from the dictionary
X = pd.DataFrame(data)
X = transformer.transform(X)

# Perform prediction
if st.button("Predict"):
    preds = np.zeros((1, 1))
    epochs = [40, 41]
    weights = [0.1, 0.9]
    for epoch, weight in zip(epochs, weights):
        weights_path = f"app//ckpt//weights_epoch_{epoch}.h5"
        standard_nn_classifier.load_weights(weights_path)  # Uncomment this line when your model is ready
        predictions = standard_nn_classifier.predict(X, verbose = 0)  # Uncomment this line when your model is ready
        preds += weight * predictions

    preds /= np.sum(weights)

    # Determine the predicted class
    predicted_class = predict_class(preds[0, 0])
    predicted_probability = preds[0, 0]

    # Display the prediction
    st.subheader("Prediction:")
    st.write(f"The predicted class is: {predicted_class}")
    st.write(f"Pr(BLL|X): {1-predicted_probability:.20f}")
    st.write(f"Pr(FSRQ|X): {predicted_probability:.20f}")
    