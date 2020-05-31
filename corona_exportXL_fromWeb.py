  
"""
EXPORT CORONA FILE TO EXCEL

AUTHOR:YANIVM

VERSION: 2.1    

"""

# import
import requests
import pandas as pd
from openpyxl import load_workbook
from datetime import datetime, timedelta,date
from threading import Timer


#intializign a timer
"""
x = datetime.today()
y = x.replace(day=x.day, hour=10, minute=30, second=0, microsecond=0) + timedelta(days=1)
delta_t = y-x
secs=delta_t.total_seconds()
"""
def exportCoronaFile():
    
    """
    export a WorldWide corona virus file ,from web to excel. 
    """
    
    #import table from HTML
    url = 'https://www.worldometers.info/coronavirus/'
    header = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
              "X-Requested-With": "XMLHttpRequest"}
    r = requests.get(url,header)
    data = pd.read_html(r.text)[0]
    
    #adding ratio column
    data['ratio'] = data['TotalDeaths']/data['TotalCases']
    
    #changing to precentage format
    def to_precent(n):
        if n==n :
            return '{:.1%}'.format(n)   #return its precentage format
        
    data['ratio'] = data['ratio'].apply(to_precent)
    
    
    
    # The 3 blocks of code ahead ment to:
    # sort the counties by 'ratio'.
    # the minor infected countries will be last
    
    #splitting data
    LargeCases = data[data['TotalCases']>100]
    MinorCases = data[data['TotalCases']<=100]
    
    #sorting
    LargeCases.sort_values(['ratio','TotalCases'] ,inplace = True, ascending = False ,na_position = 'last')
    MinorCases.sort_values(['ratio','TotalCases'] ,inplace = True, ascending = False ,na_position = 'last')
    
    #concatination
    data = pd.concat((LargeCases,MinorCases))
    
    #changing columns names
    data.columns = ['Country', 'TotalCases', 'NewCases', 'TotalDeaths', 'NewDeaths','TotalRecovered', 'ActiveCases', 'Critical', 'Tot Cases/1M pop','Deaths/1M pop', 'TotalTests', 'Tests/ 1M pop', 'ratio']
    
    #set Country as index
    data.set_index(keys = data['Country'] ,inplace = True)
    data.drop('Country',axis=1,inplace=True)
    
    #initialize date to 'today'
    today = date.today()
    today = str(today)
    
    #export to excel:
    path = r'C:\Users\97250\Desktop\studied\R ,python\corona\coronaWorldWide.xlsx'
    book = load_workbook(path)
    writer = pd.ExcelWriter(path, engine = 'openpyxl')
    writer.book = book
    data.to_excel(writer ,sheet_name = today) #the sheet name will be the str(current date)
    writer.save()
    writer.close()

exportCoronaFile()
# activate chronologic action
"""
t = Timer(secs, exportCoronaFile)
t.start()
"""