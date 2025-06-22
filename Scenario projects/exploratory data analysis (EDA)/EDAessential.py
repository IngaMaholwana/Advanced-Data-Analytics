import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

companies = pd.read_csv("Unicorn_Companies.csv")

companies.head(10)

companies.size

companies.shape

companies.info()
companies.describe()

# Convert 'Date Joined' to datetime format
companies["Date Joined"] = pd.to_datetime(companies["Date Joined"])

companies.info()
#created a year joined column
companies["Year Joined"] = companies["Date Joined"].dt.year
companies.head()

# a sample of the data
companies_sample = companies.sample(n = 50, random_state = 42)

#visualize the sample data  
companies_sample["years_till_unicorn"] = companies_sample["Year Joined"] - companies_sample["Year Founded"]

# Group the data by `Industry`. For each industry, get the max value in the `years_till_unicorn` column.
grouped = (companies_sample[["Industry", "years_till_unicorn"]]
           .groupby("Industry")
           .max()
           .sort_values(by="years_till_unicorn")
          )
grouped
# print(grouped)