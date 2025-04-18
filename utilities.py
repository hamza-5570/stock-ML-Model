import pandas as pd
from datetime import date
from fastapi.responses import StreamingResponse
import io


# Download forecast into client machine
def get_forecast(forecast:pd.DataFrame):
    # Convert DataFrame to CSV in-memory
    output = io.StringIO()
    forecast.to_csv(output, index=False)
    output.seek(0)
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv",headers={"Content-Disposition": f"attachment; filename=sku_{forecast.sku.iloc[0]}_forecast.csv"})



# Calculate reorder point
def calculate_reorder_point(forecast: pd.Series, lead_time: int, safety_stock: int):
    average_daily_demand = sum(forecast) / len(forecast)
    reorder_point = (average_daily_demand * lead_time) + safety_stock
    return reorder_point.round()



# Function to simulate future inventory and check reorder alerts
def check_reorder_alert(current_inventory:int, forecast:pd.Series, reorder_point:int):
    for day in forecast:
        current_inventory = current_inventory - forecast[day]
        if current_inventory <= reorder_point:
            return f"Reorder Alert: Reorder needed by Day {day}."
    return "Stock level is sufficient."



# Function to check for overstock warnings
def check_overstock_warning(current_inventory:int, forecast_total:int):
    if current_inventory > forecast_total:
        return "Overstock Warning: Current inventory exceeds forecasted demand."
    else:
        return "No overstock warning."



# Function to get the filtered forecast based on the input dates
def get_filtered_total_forecast(forecast:pd.DataFrame, start_day:date, end_day:date):
    # Get the filtered forecast
    start_day = pd.to_datetime(start_day)
    end_day = pd.to_datetime(end_day)
    filtered_forecast = forecast[(forecast.date>=start_day) & (forecast.date<=end_day)]
    return f"Total Demand for SKU: {filtered_forecast.sku.iloc[0]} for {filtered_forecast.shape[0]} days ({start_day.strftime('%Y-%m-%d')}----{end_day.strftime('%Y-%m-%d')}): {filtered_forecast.forecast.sum()}"