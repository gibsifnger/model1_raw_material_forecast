#pip install yfinance
import yfinance as yf

df = yf.download("ZC=F", interval="1mo", period="max", auto_adjust=False)
df = df.reset_index()
df.to_csv("ZC_F_monthly.csv", index=False, encoding="utf-8-sig")
print(df.tail())