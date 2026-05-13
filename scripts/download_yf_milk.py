import pandas as pd

url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=WPU016"

milk = pd.read_csv(url)

milk = milk.rename(columns={
    "observation_date": "Date",
    "WPU016": "milk_price"
})

milk["Date"] = pd.to_datetime(milk["Date"])
milk["milk_price"] = pd.to_numeric(milk["milk_price"], errors="coerce")

milk = milk.dropna(subset=["milk_price"]).reset_index(drop=True)

milk.to_csv("milk_price.csv", index=False, encoding="utf-8-sig")

print(milk.head())
print("saved: milk_price.csv")