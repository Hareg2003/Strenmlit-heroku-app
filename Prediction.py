import streamlit as st
from pickle import load
import numpy as np

st.title("Diamond Price Prediction")
st.header("Don't Worry, you don't need to give all the features of Diamond, I only want two features of Diamond to do price prediction  ;) ")

regressor_rf = load(open('pkl/model_rf.pkl', 'rb'))
scaler = load(open('pkl/trans_scale.pkl', 'rb'))

clarity_encoder = {'I1':1, 'SI2':2, 'SI1':3, 'VS2':4, 'VS1':5, 'VVS2':6, 'VVS1':7, 'IF':8}

carat = st.number_input('Carat range 0.2-5.01')

clarity = st.selectbox(
     'How should be the clarity of Diamond?',
     ('I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'))

num_scal=scaler.transform([[carat]])

cat_encod=np.array([clarity_encoder[clarity]])

if st.button("Predict")==True:
    prediction=regressor_rf.predict(np.concatenate((num_scal.flatten(), cat_encod), axis=None).reshape(1,-1)).item()
    st.write(prediction)
    st.balloons()