from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from xgboost import XGBRegressor
from mlforecast import MLForecast
import numpy as np
import io
from pydantic import BaseModel
from typing import Optional
from pytz import timezone
from datetime import date
from utilities import calculate_reorder_point, check_reorder_alert, check_overstock_warning, get_filtered_total_forecast, get_forecast
from authentication import verify_token
from subcat_forecast import infer_gender_age_from_title, assign_subcategory, get_weighted_subcategory_forecast, forecast_horizon


# Get current date format (US Central timezone)
central_time = pd.Timestamp.now(timezone('US/Central'))

# Format the current date as required
current_date = central_time.strftime('%d-%m-%y')

# Add one day
next_day = (central_time + pd.Timedelta(days=1)).strftime('%d-%m-%y')


# Set random seed for reproducibility
np.random.seed(43)

# In-memory storage for uploaded product data
product_data = None

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Or set your frontend domain here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Authorization"],  # 👈 This allows Authorization to be sent
)

# Input validation
class SKURequest(BaseModel):
    sku: str
    product_title: str
    category: str
    subcategory: str
    price: float
    material: str
    gender_age: str
    lead_time: int = 7
    safety_stock: int = 22
    current_inventory: Optional[int] = None
    start_day: Optional[date] = None
    end_day: Optional[date] = None



# Function to load data
def load_product_data(file: UploadFile):
    global product_data
    try:
        df = pd.read_csv(io.StringIO(file.file.read().decode("utf-8")), parse_dates=["SaleDate"])
        cols = ["ProductTitle", "Category", "Subcategory", "Material", "Gender_Age"]
        df[cols] = df[cols].astype(str).map(str.lower).map(str.strip)
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        product_data = df
        # Lowercase columns names
        product_data.columns = product_data.columns.str.lower().str.strip()

    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid CSV format")



# Preprocess data based on sku
def preprocess_data_for_sku(sku:str, data:pd.DataFrame):
    temp_df = data[data.sku==sku].sort_values("saledate")
    temp_df = temp_df.groupby(["saledate"]).unitssold.sum().to_frame().reset_index()
    master_date = pd.DataFrame(pd.date_range(start=temp_df.saledate.iloc[0], end=temp_df.saledate.iloc[-1], freq="D"), columns=["date"])
    m = pd.merge(master_date, temp_df, left_on="date", right_on="saledate", how="left").drop("saledate", axis=1)
    m.unitssold = m.unitssold.fillna(0)
    m["unique_id"] = sku
    m.columns = ["ds", "y", "unique_id"]
    return m



# Make future forecast based on sku
def make_sku_based_forecast(sku:str, product_data:pd.DataFrame):

    # Preprocess data
    # Last date to retrieve current_inventory
    global last_date
    data = preprocess_data_for_sku(sku, product_data)
    last_date = data.ds.iloc[-1]
    
    # Create future forecast data
    next_day = data.ds.iloc[-1] + pd.Timedelta(days=1)
    future_df = pd.DataFrame(pd.date_range(start=next_day, freq="d", periods=forecast_horizon), columns=["ds"])
    future_df["unique_id"] = sku
    
    # Build the model
    fcst = MLForecast(
        models=XGBRegressor(random_state=42),
        freq="d",
        lags=[30],
        date_features=["day", "month"])
    
    
    fcst.fit(data, static_features=[])
    pred = fcst.predict(h=forecast_horizon)
    pred = pred.drop("unique_id", axis=1)
    pred.columns = ["date", "forecast"]
    pred.forecast = pred.forecast.apply(lambda x: 0 if x<0.5 else x).round().astype("int")
    pred["sku"] = sku
    return pred



# Endpoint for check the status of the server
@app.get("/", include_in_schema=False)
async def health_check(token: bool = Depends(verify_token)):
    return {"status": "Healthy & Running"}



# Endpoint for uploading the data
@app.post("/upload-train-data/")
async def upload_csv(file: UploadFile = File(None), token: bool = Depends(verify_token)):
    if file is None:
        raise HTTPException(status_code=400, detail="Training data not uploaded")

    try:
        load_product_data(file)
        return {"message": "Train data uploaded successfully"}
    except Exception as e:
        return {"message": f"Error processing file: {str(e)}"}



# Global variable to store the latest forecast
forecast = None


# Endpoint to forecast demand
@app.post("/make-forecast/")
async def predict_sku(sku_request: SKURequest, token: bool = Depends(verify_token)):

    # Make it available outside of this function for caching
    global forecast
    global product_data

    if product_data is None:
        raise HTTPException(status_code=400, detail="No train data uploaded. Please upload a CSV first.")

    # Convert SKURequest to DataFrame
    new_sku = pd.DataFrame([sku_request.dict()])

    # Lowercase all the object columns
    obj_columns = new_sku.select_dtypes("object").columns[:-2]
    new_sku[obj_columns] = new_sku[obj_columns].apply(lambda x:x.str.lower().str.strip())
    new_sku.columns = new_sku.columns.str.lower().str.strip()
    sku_value = new_sku["sku"].iloc[0]

    # Check sales history
    sales_history = product_data[product_data["sku"] == sku_value]["unitssold"].count()


    # Full sku based forecasting
    if sales_history >= 90:
        forecast = make_sku_based_forecast(sku_value, product_data)

        # Calculate reorder point
        reorder_point = calculate_reorder_point(forecast.forecast, new_sku.lead_time.iloc[0], new_sku.safety_stock.iloc[0])

        # Insert reorder point into forecast df
        forecast["reorder_point"] = round(reorder_point)

        # Reorganize the columns
        forecast = forecast[["sku", "date", "forecast", "reorder_point"]]

       
        # Do reorder alert warning
        try:
            # Get current inventory from the training data
            current_inventory = product_data[(product_data.sku==sku_value) & (product_data.saledate==last_date)].currentinventory.iloc[0]
            reorder_alert = check_reorder_alert(current_inventory, forecast.forecast, reorder_point)
        except:
            raise HTTPException(status_code=404, detail="No current inventory in the training data")

        # Do overstock warning
        overstock_warning = check_overstock_warning(current_inventory, forecast.forecast.sum())

        # Do filtered forecast
        try:
            filtered_forecast = get_filtered_total_forecast(forecast, new_sku.start_day.iloc[0], new_sku.end_day.iloc[0])
            return filtered_forecast, reorder_alert, overstock_warning, forecast.to_dict(orient="records")
        except:
            return reorder_alert, overstock_warning, forecast.to_dict(orient="records")



    elif 30 <= sales_history < 90:
        subcategory = assign_subcategory(new_sku, product_data)
        sku_forecast = make_sku_based_forecast(sku_value, product_data)
        subcategory_forecast = get_weighted_subcategory_forecast(new_sku, product_data, subcategory, forecast_horizon)

        # Weighted transition to SKU-level forecasting
        # Combination of sku-based and subcategory-based
        weight = sales_history / 90

        forecast = [(weight * sku_f) + ((1 - weight) * subcat_f) for sku_f, subcat_f in zip(sku_forecast, subcategory_forecast)]
        forecast["sku"] = sku_value

        # Calculate reorder point
        reorder_point = calculate_reorder_point(forecast.forecast, new_sku.lead_time.iloc[0], new_sku.safety_stock.iloc[0])

        # Insert reorder point into forecast df
        forecast["reorder_point"] = round(reorder_point)

        # Insert date
        date = pd.date_range(start=next_day, periods=forecast_horizon, freq="D")
        forecast["date"] = date

        # Reorganize the columns
        forecast = forecast[["sku", "date", "forecast", "reorder_point"]]

        # Do reorder alert warning
        try:
            # Get current inventory from the training data
            current_inventory = product_data[(product_data.sku==sku_value) & (product_data.saledate==last_date)].currentinventory.iloc[0]
            reorder_alert = check_reorder_alert(current_inventory, forecast.forecast, reorder_point)
        except:
            raise HTTPException(status_code=404, detail="No current inventory in the training data")

        # Do overstock warning
        overstock_warning = check_overstock_warning(current_inventory, forecast.forecast.sum())

        # Do filtered forecast
        try:
            filtered_forecast = get_filtered_total_forecast(forecast, new_sku.start_day.iloc[0], new_sku.end_day.iloc[0])
            return filtered_forecast, reorder_alert, overstock_warning, forecast.to_dict(orient="records")
        except:
            return reorder_alert, overstock_warning, forecast.to_dict(orient="records")


    else:
        # Only subcategory-based forecasting
        subcategory = assign_subcategory(new_sku, product_data)

        forecast = get_weighted_subcategory_forecast(new_sku, product_data, subcategory)
        forecast["sku"] = sku_value

        # Calculate reorder point
        reorder_point = calculate_reorder_point(forecast.forecast, new_sku.lead_time.iloc[0], new_sku.safety_stock.iloc[0])

        # Insert reorder point into forecast df
        forecast["reorder_point"] = round(reorder_point)

        # Insert date
        date = pd.date_range(start=next_day, periods=forecast_horizon, freq="D")
        forecast["date"] = date

        # Reorganize the columns
        forecast = forecast[["sku", "date", "forecast", "reorder_point"]]


        # Do reorder alert
        try:
            # Get current inventory
            current_inventory = new_sku.current_inventory.iloc[0]
            reorder_alert = check_reorder_alert(current_inventory, forecast.forecast, reorder_point)
        except Exception as e:
            raise HTTPException(status_code=404, detail="No current inventory input from user")

        # Do overstock warning
        overstock_warning = check_overstock_warning(current_inventory, forecast.forecast.sum())

        # Do filtered forecast
        try:
            filtered_forecast = get_filtered_total_forecast(forecast, new_sku.start_day.iloc[0], new_sku.end_day.iloc[0])
            return filtered_forecast, reorder_alert, overstock_warning, forecast.to_dict(orient="records")
        except:
            return reorder_alert, overstock_warning, forecast.to_dict(orient="records")



# Endpoint to download forecast
@app.get("/download-forecast/")
async def download_forecast(token: bool = Depends(verify_token)):
    if forecast is not None:
        return get_forecast(forecast)
    raise HTTPException(status_code=404, detail="No forecast available, please make a forecast first")
    
