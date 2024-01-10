import pandas as pd

def wrangle(data, loan_device_id, unit, lag="4H"):
    df=data[['loan_device_id','upper_threshold','lower_threshold','value','unit']]
    mask = (df['unit'] == unit) & (df['loan_device_id'] == loan_device_id)
    new_df = df.loc[mask]
    y = new_df["value"]
    y = y.resample(lag).mean().fillna(method='ffill')
    return y 