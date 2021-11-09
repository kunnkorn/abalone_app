import streamlit as st
import pandas as pd

st.write("""# My First Web Application 
Let's enjoy **Data Science** project !!!
""")

st.sidebar.header('User Input')
st.sidebar.subheader('Please enter your data: ')

V_Sex = st.sidebar.radio('Sex' , ['Male','Female', 'Infant'])
v_length = st.sidebar.slider('Length' , min_value = 0.0, max_value = 1.0 , value = 0.5)
v_diameter = st.sidebar.slider('Diameter' , min_value = 0.0, max_value = 1.0 , value = 0.4)
v_height = st.sidebar.slider('Height' , min_value = 0.0, max_value = 1.0 , value = 0.1)
v_whole = st.sidebar.slider('Whole Weight' , min_value = 0.0, max_value = 3.0 , value = 0.8)
v_shucked = st.sidebar.slider('Shucked Weight' , min_value = 0.0, max_value = 2.0 , value = 0.3)
v_viscera = st.sidebar.slider('Viscera Weight' , min_value = 0.0, max_value = 1.0 , value = 0.2)
v_shell = st.sidebar.slider('Shell Weight' , min_value = 0.0, max_value = 2.0 , value = 0.2)

#Change the value of Sex to be {M,F,I} as stored in trained set
if V_Sex == 'Male': V_Sex = 'M'
elif V_Sex == 'Female': V_Sex='F'
else: V_Sex = 'I'


#Store User Input data in a dictionary
data = {'Sex': V_Sex, 
'Length': v_length , 
'Diameter': v_diameter , 
'Height': v_height , 
'Whole_weight': v_whole, 
'Shucked_weight': v_shucked,
'Viscera_weight': v_viscera,
'Shell_weight': v_shell
}

df = pd.DataFrame(data , index=[0])


#Main Panel
st.header('Application of Abalone\'s Age Predection: ')
st.subheader('USer Input: ')

st.write(df)

# Combines user input data with sample dataset
# The sample data contains unique values for each nominal features
# This will be used for the One-hot encoding
data_sample = pd.read_csv('abalone_sample_data.csv')
df = pd.concat([df, data_sample],axis=0)
# st.write(df)

#One-hot encoding for nominal features
cat_data = pd.get_dummies(df[['Sex']])
# st.write(cat_data)

#Combine all transformed features together
X_new = pd.concat([cat_data, df], axis=1)

# Select only the first row (the user input data)
X_new = X_new[:1] 

#Drop un-used feature
X_new = X_new.drop(columns=['Sex'])
#Show the X_new data frame on the screen
st.subheader('Pre-Processed Input:')
st.write(X_new)

import pickle
# -- Reads the saved normalization model
load_nor = pickle.load(open('normalization.pkl', 'rb'))
#Apply the normalization model to new data
X_new = load_nor.transform(X_new)
#Show the X_new data frame on the screen
st.subheader('Normalized Input:')
st.write(X_new)


# -- Reads the saved classification model
load_knn = pickle.load(open('best_knn.pkl', 'rb'))
# Apply model for prediction
prediction = load_knn.predict(X_new)
#Show the prediction result on the screen
st.subheader('Prediction:')
st.write(prediction)
