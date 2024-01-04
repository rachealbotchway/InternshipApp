import streamlit as st
import pickle
#st.write('hello') #testing streamlit

date=st.date_input('enter date needed to predict',value=None)#input for taking a date from user, none for removing automated date
if date:  
    # Load the trained model from a pickle file
    with open("best_model_new.pkl", "rb") as file:
        model = pickle.load(file)
 
    
    #st.write(date)
    # Make predictions
    #fit_model = model.fit()
    forecast_steps = 10  # Number of steps to forecast
    predicted_values = model.forecast(date)[-5:]  #taking the last prediction before the set date
    st.write(predicted_values)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(predicted_values, label='Actual Values')
    plt.xticks(rotation=30, ha='right')
    st.pyplot(fig)

    
