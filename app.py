# libraries
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from functions import * 

st.header("Loan Device Temperature and Current Predictor")
#input for taking a date from user, none for removing automated date
uploaded_file = st.file_uploader("Load the Loan device csv", "csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['datetime'], index_col=['datetime'])

    df.dropna(subset = ['loan_device_id'], inplace=True)

    device_ids = tuple(df['loan_device_id'].astype(int).unique())

    device_id = st.selectbox(
        "Loan Device ID",
        device_ids,
        index=0)
    
    units = tuple(df[df['loan_device_id'] == device_id]['unit'].unique())
    unit = st.selectbox(
        'Unit of measurement?',
        units,
        index=0)

    lag = st.selectbox(
        'How would you like the lag to be? (optional, default=4H)',
        ('1H', '2H', '4H', '6H', '12H', "1D"),
        index=2)
    
    date = st.date_input('Enter date needed to predict', value=None)

    data = wrangle(df, device_id, unit, lag)

    if date and len(data) > 0: 
        
        # Set the order for the ARIMA model using p,d,q
        p, d, q = 6, 0, 5  #values for p, d, q

        # Fit ARIMA model to the data
        model = ARIMA(data, order=(p, d, q))
        model_fit = model.fit()
    
        # Make predictions
        #fit_model = model.fit()
        forecast_steps = 10  # Number of steps to forecast
        #taking the last prediction before the set date
        predicted_values = model_fit.forecast(date)[-5:]  
        st.write(predicted_values)

        fig, ax = plt.subplots()
        ax.plot(predicted_values, label='Actual Values')
        plt.xticks(rotation=30, ha='right')
        st.pyplot(fig)

    
