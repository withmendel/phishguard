import pandas as pd

print("=== MeAJOR (Email) ===")
email_df = pd.read_csv("data/meajor.csv")
print(f"Shape: {email_df.shape}")
print(f"Columns: {list(email_df.columns)[:10]}...")
print(f"Label distribution:\n{email_df['label'].value_counts()}")
print(f"Sample body:\n{email_df['body'].iloc[0][:300]}")

print("\n=== PhiUSIIL (URL) ===")
url_df = pd.read_csv("data/PhiUSIIL_Phishing_URL_Dataset.csv")
print(f"Shape: {url_df.shape}")
print(f"Columns: {list(url_df.columns)[:10]}...")
print(f"Label distribution:\n{url_df['label'].value_counts()}")
