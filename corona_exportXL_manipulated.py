"""
Created on Apr 20 17:12:30 2020

@author: Yaniv Maimon

version : 1.2

"""

# imports
import pandas as pd
from openpyxl import load_workbook

# import EXCEL file all sheets
path = r'C:\Users\97250\Desktop\studied\R ,python\corona\coronaWorldWide.xlsx'
x = pd.read_excel(path, sheet_name=None)

# Unified XL sheets only
x_indeces = [i for i in x][5:]

# initialize First sheet
first_sheet = x_indeces[0]

# initialize file
x[first_sheet].to_excel(r'C:\Users\97250\Desktop\studied\R ,python\corona\coronaWorldWide_afterTransmission_ver2.xlsx',sheet_name = first_sheet)

path = r'C:\Users\97250\Desktop\studied\R ,python\corona\coronaWorldWide_afterTransmission_ver2.xlsx'
book = load_workbook(path)

#    manipulating data

for i in x_indeces:

    # Removing additional signs and convert to floats.
    x[i]['NewCases']   = x[i]['NewCases'].apply(lambda x: float(str(x).replace("+","").replace(',','')))
    x[i]['NewDeaths']  = x[i]['NewDeaths'].apply(lambda x: float(str(x).replace("+","").replace(',','')))
    x[i]['ratio']      = x[i]['ratio'].apply(lambda x: float(str(x).replace("%",""))/100)

    #adding new column
    x[i]['Tests/Cases']        = round(x[i]['TotalTests']/x[i]['TotalCases'],2)

    # Filling NA's
    x[i].fillna(value = 0,inplace =True)

    # Unified colum form
    x[i].columns = ['Country', 'Total Cases', 'New Cases', 'Total Deaths', 'New Deaths',
                   'Total Recovered', 'Active Cases', 'Critical', 'Cases/1M pop',
                   'Deaths/1M pop', 'Total Tests', 'Tests/1M pop',
                    'Deaths/Cases','Tests/Cases']

    #set index
    x[i].set_index('Country',inplace=True)

    # Removing 'world'
    x[i].drop('World',inplace=True)

    #removing 'Total:'
    try:
        x[i].drop('Total:',inplace=True)
    except:
        pass

    #export to excel as additional sheets:
    writer = pd.ExcelWriter(path, engine = 'openpyxl')
    writer.book = book
    x[i].to_excel(writer ,sheet_name = i) #the sheet name will be the date


writer.save()
writer.close()
