from cleaning import get_clean_data
import matplotlib
data = get_clean_data()
print(data.info)
print(set(data.columns))