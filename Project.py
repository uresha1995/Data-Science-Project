#Downlaod relevant libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Downlaod data set
data = pd.read_csv(r"C:\Users\uresha\Dropbox\PC\Desktop\UH\Project\Data\Death and Incidence.csv")

#Check first few raws
print(data.head())

#EDA

#Shape of the dataset
print(f'Death : {data.shape}')

#Check types of the data and values
print(data.info())

#Missing values
print(data.isnull().sum())

#Summary statistics
print(data.describe())

# Unique countries in each dataset
#print(f"Data : {data['location_name'].unique()}")


