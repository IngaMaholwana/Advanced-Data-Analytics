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

# Create bar plot
# with Industry column as the categories of the bars
# and the difference in years between Year Joined column and Year Founded column as the heights of the bars


plt.bar(grouped.index, grouped["years_till_unicorn"])

# Set titling


plt.title("Bar plot of maximum years taken by company to become unicorn per industry (from sample)")

# Set x-axis label


plt.xlabel("Industry")

# Set y-axis label


plt.ylabel("Maximum number of years")

# Rotate labels on the x-axis as a way to avoid overlap in the positions of the text  


plt.xticks(rotation=45, horizontalalignment='right')
#save it
# plt.savefig("unicorn_years_by_industry.png", dpi=300)

# Display the plot

plt.show()

# Create a column representing company valuation as numeric data

# Create new column
companies_sample['valuation_billions'] = companies_sample['Valuation']
# Remove the '$' from each value
companies_sample['valuation_billions'] = companies_sample['valuation_billions'].str.replace('$', '')
# Remove the 'B' from each value
companies_sample['valuation_billions'] = companies_sample['valuation_billions'].str.replace('B', '')
# Convert column to type int
companies_sample['valuation_billions'] = companies_sample['valuation_billions'].astype('int')
companies_sample.head()

# Prepare data for modeling
grouped = (companies_sample[["Industry", "valuation_billions"]]
           .groupby("Industry")
           .max()
           .sort_values(by="valuation_billions")
          )
grouped

# Create bar plot
# with Industry column as the categories of the bars
# and new valuation column as the heights of the bars


plt.bar(grouped.index, grouped["valuation_billions"])

# Set title


plt.title("Bar plot of maximum unicorn company valuation per industry (from sample)")

# Set x-axis label


plt.xlabel("Industry")

# Set y-axis label


plt.ylabel("Maximum valuation in billions of dollars")

# Rotate labels on the x-axis as a way to avoid overlap in the positions of the text  


plt.xticks(rotation=45, horizontalalignment='right')

# Display the plot


plt.show()