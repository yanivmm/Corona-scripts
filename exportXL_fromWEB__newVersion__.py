
"""
EXPORT CORONA FILE TO EXCEL

AUTHOR:YANIVM

VERSION: 2.1

"""


# imports
import requests
import pandas as pd
from datetime import date
from openpyxl import load_workbook


url = 'https://www.worldometers.info/coronavirus/'
header = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
          "X-Requested-With": "XMLHttpRequest"}
r = requests.get(url,header)
data = pd.read_html(r.text)[0]



# Removing additional signs and convert to floats.
data['NewCases']   = data['NewCases'].apply(lambda x: float(str(x).replace("+","").replace(',','')))
data['NewDeaths']  = data['NewDeaths'].apply(lambda x: float(str(x).replace("+","").replace(',','')))

#adding new columns
data['Deaths/Cases']       = round(data['TotalDeaths']/data['TotalCases'],2)
data['Tests/Cases']        = round(data['TotalTests']/data['TotalCases'],2)

# Fill NA's
data.fillna(value = 0,inplace =True)

#drop columns
data.drop(['#','Population'] , axis=1, inplace = True)

# Unified colum form
data.columns = ['Country', 'Total Cases', 'New Cases', 'Total Deaths', 'New Deaths',
               'Total Recovered', 'Active Cases', 'Critical', 'Cases/1M pop',
               'Deaths/1M pop', 'Total Tests', 'Tests/1M pop',
                'Deaths/Cases','Tests/Cases']

# set country as index index
data.set_index('Country',inplace=True)

# Removing 'world' row
data.drop(['World','Total:'],inplace=True)



###                export to XL file


today = str(date.today())


path = r'C:\Users\97250\Desktop\studied\R ,python\corona\coronaWorldWide_afterTransmission_ver2.xlsx'
book = load_workbook(path)
writer = pd.ExcelWriter(path, engine = 'openpyxl')
writer.book = book
data.to_excel(writer ,sheet_name = today) #the sheet name will be the str(current date)
writer.save()
writer.close()
