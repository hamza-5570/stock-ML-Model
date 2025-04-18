import pandas as pd
import numpy as np
from rapidfuzz import fuzz


# Define forecast horizon
forecast_horizon = 90


def infer_gender_age_from_title(product_title:str):
    """
    Infers the gender/age category from the product title.
    Example mappings can be expanded based on business rules.
    """
    if any(keyword in product_title.lower() for keyword in ["men", "male", "guy"]):
        return "Men"
    elif any(keyword in product_title.lower() for keyword in ["women", "female", "lady"]):
        return "Women"
    elif any(keyword in product_title.lower() for keyword in ["boy", "toddler boy", "baby boy"]):
        return "Boys"
    elif any(keyword in product_title.lower() for keyword in ["girl", "toddler girl", "baby girl"]):
        return "Girls"
    else:
        return "Unisex"



# Function to assign Subcategory to New SKUs
def assign_subcategory(new_sku:pd.DataFrame, product_data:pd.DataFrame):
    """
    Assigns a subcategory to a new SKU using fuzzy matching, category-based filtering, and price similarity.
    If Gender_Age is missing, it attempts to infer from the ProductTitle.
    """
    # Step 1: Exact SKU Title Match
    exact_match = product_data[product_data["producttitle"] == new_sku["product_title"].iloc[0]]
    if not exact_match.empty:
        return exact_match["subcategory"].values[0]

    # Step 2: Fuzzy Title Matching
    product_data["Title_Similarity"] = product_data["producttitle"].apply(lambda x: fuzz.ratio(x, new_sku["product_title"].iloc[0]))
    similar_titles = product_data[product_data["Title_Similarity"] > 85]  # Threshold for similarity
    # Get the best one title with highest similarity score
    similar_titles = similar_titles.sort_values("Title_Similarity", ascending=False)

    if not similar_titles.empty:
        return similar_titles["subcategory"].iloc[0]

    # Step 3: Handle Missing Gender_Age by Inferring from ProductTitle
    if "gender_age" not in new_sku.index or pd.isna(new_sku.get("gender_age")):
        inferred_gender_age = infer_gender_age_from_title(new_sku["product_title"].iloc[0])
    else:
        inferred_gender_age = new_sku["gender_age"]

    # Step 4: Filter by Inferred Gender_Age
    filtered_data = product_data[product_data["gender_age"] == inferred_gender_age]

    # Step 5: Filter by Category and Price
    category_skus = filtered_data[filtered_data["category"] == new_sku["category"].iloc[0]]
    price_filtered = category_skus[
        (category_skus["price"] >= new_sku["price"].iloc[0] * 0.8) & 
        (category_skus["price"] <= new_sku["price"].iloc[0] * 1.2)
    ]

    # Step 6: Assign Most Frequent Subcategory if No Match Found
    if not price_filtered.empty:
        return price_filtered["subcategory"].mode()[0]

    return product_data["subcategory"].mode()[0] if not product_data.empty else "General - Needs Review"




# Function to get Weighted Subcategory Forecast (Now with Time-Series Variation)
def get_weighted_subcategory_forecast(new_sku:pd.DataFrame, product_data:pd.DataFrame, subcategory:str):
    """
    Estimates demand for a new SKU based on subcategory-level forecasting.
    This version introduces time-series variation to prevent identical forecasts across timestamps.
    """

    # Making the noise reproducible
    np.random.seed(43)

    # Step 1: Filter historical SKUs in the same subcategory
    subcategory_data = product_data[product_data["subcategory"] == subcategory]

    if subcategory_data.empty:
        return None  # No subcategory-level data available

    # Step 2: Calculate SKU Contribution Weights
    total_units_sold = subcategory_data["unitssold"].sum()
    subcategory_data["Sales_Contribution"] = subcategory_data["unitssold"] / total_units_sold if total_units_sold > 0 else 0

    # Step 3: Extract Historical Trends (Moving Average for Forecasting)
    subcategory_data["Rolling_Trend"] = subcategory_data["unitssold"].rolling(window=7, min_periods=1).mean()

    # Step 4: Compute Weighted Forecast for Each Time Step
    forecast_series = []
    base_forecast = (subcategory_data["Sales_Contribution"] * subcategory_data["Rolling_Trend"]).sum()

    for t in range(forecast_horizon):
        # Introduce variation using past trend shifts
        trend_factor = np.sin(2 * np.pi * t / forecast_horizon) * 0.1  # Simulated seasonality pattern
        noise = np.random.normal(0, 0.5)  # Adding a small amount of noise for variation

        forecast_t = base_forecast * (1 + trend_factor + noise)
        forecast_series.append(forecast_t)

    # Step 5: Apply Price Filtering (Optional)
    price_filtered = subcategory_data[
        (subcategory_data["price"] >= new_sku["price"].iloc[0] * 0.8) & 
        (subcategory_data["price"] <= new_sku["price"].iloc[0] * 1.2)
    ]

    if not price_filtered.empty:
        forecast_series = []
        base_forecast = (price_filtered["Sales_Contribution"] * price_filtered["Rolling_Trend"]).sum()

        for t in range(forecast_horizon):
            trend_factor = np.sin(2 * np.pi * t / forecast_horizon) * 0.1
            noise = np.random.normal(0, 0.5)

            forecast_t = base_forecast * (1 + trend_factor + noise)
            forecast_series.append(forecast_t)
    forecast_series = pd.DataFrame(forecast_series, columns=["forecast"])
    forecast_series.forecast = forecast_series.forecast.apply(lambda x: 0 if x<0.5 else x).astype("int")
    return forecast_series