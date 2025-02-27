import uuid
import pandas as pd
import random
from datetime import datetime, timedelta

# Function to generate synthetic eCommerce transactions data
def generate_synthetic_data(rows=10000):
    categories = ["Electronics", "Clothing", "Home & Kitchen", "Books", "Toys"]
    payment_methods = ["Credit Card", "PayPal", "Debit Card", "Gift Card"]
    statuses = ["Delivered", "Shipped", "Processing", "Cancelled"]
    
    data = {
        "Order ID": [str(uuid.uuid4()) for _ in range(rows)],
        "Customer ID": [str(uuid.uuid4()) for _ in range(rows)],
        "Product ID": [str(uuid.uuid4()) for _ in range(rows)],
        "Product Category": [random.choice(categories) for _ in range(rows)],
        "Price": [round(random.uniform(10, 500), 2) for _ in range(rows)],
        "Quantity Ordered": [random.randint(1, 5) for _ in range(rows)],
        "Discount Applied": [round(random.uniform(0, 0.3), 2) for _ in range(rows)],
        "Order Date": [(datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d') for _ in range(rows)],
        "Shipping Date": [(datetime.now() - timedelta(days=random.randint(0, 360))).strftime('%Y-%m-%d') for _ in range(rows)],
        "Delivery Status": [random.choice(statuses) for _ in range(rows)],
        "Payment Method": [random.choice(payment_methods) for _ in range(rows)],
        "Total Order Value": [0] * rows,  # Will be calculated later
        "Customer Age": [random.randint(18, 65) for _ in range(rows)],
        "Customer Gender": [random.choice(["Male", "Female", "Other"]) for _ in range(rows)],
        "Customer Location": [random.choice(["USA", "Canada", "UK", "Germany", "France"]) for _ in range(rows)],
        "Return Status": [random.choice(["Returned", "Not Returned"]) for _ in range(rows)],
        "Warehouse ID": [str(uuid.uuid4()) for _ in range(rows)],
        "Shipment Partner": [random.choice(["DHL", "FedEx", "UPS", "USPS"]) for _ in range(rows)],
    }
    
    df = pd.DataFrame(data)
    df["Discounted Price"] = df["Price"] - (df["Price"] * df["Discount Applied"])
    df["Total Order Value"] = df["Discounted Price"] * df["Quantity Ordered"]
    return df

if __name__ == "__main__":
    data = generate_synthetic_data(1000)

    data.to_csv("data/transactions.csv")