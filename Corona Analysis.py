
"""
Corona(COVID-19) Analysis ML (random forest)

AUTHOR:YANIVM

VERSION: 2.1

"""


# imports
import requests
import pandas as pd
import numpy as np

url = r'https://www.worldometers.info/coronavirus/'
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

# copy
coronaData = data.copy()[['Cases/1M pop', 'Deaths/Cases']]#','Tests/1M pop','Tests/Cases','Deaths/1M pop'


#           ctaholicData


url = r'https://en.wikipedia.org/wiki/Catholic_Church_by_country'
data = pd.read_html(url)[2]

# drop columns
data.drop(['Total population','Catholic total'] , axis=1, inplace = True)

#columns name
data.columns = ['Country','%']

# arrange Country column
data['Country'] = data['Country'].apply(lambda x: x[:-10]) 

#precentage converting function
def precentage(x):
    x=str(x)
    
    if x.count('-')>1:  # if multiple data 
            l = [int(float(i.split('%')[0])) for i in x.split('-')]
            return np.mean(l)  #makes average of datas
    else:
        return int(float(x.split('%')[0]))  #return int data type


# fix single row before applying precentage
data.iloc[29]['%'] = 0

#arrange % catholic columns
data['%'] = data['%'].apply(precentage)

# set country as index index
data.set_index('Country',inplace=True)
data.columns = ['Catholic%']

# Removing 'Total' row
data.drop('',inplace=True)

# assign to new variable
catholicData = data.copy()



#           GDPData


url = r'https://en.wikipedia.org/wiki/List_of_countries_by_GDP_%28PPP%29_per_capita'
data = pd.read_html(url)[2]

# arrange columns
data.columns = ['Rank','Country','GDPC']
data.drop('Rank',axis=1,inplace=True)

# set country as index index
data.set_index('Country',inplace=True)

# copy
GDPData = data.copy()



#           IsalamData


url = r'https://en.wikipedia.org/wiki/Islam_by_country'
data = pd.read_html(url)[2]

#drop countries
data.drop(['Total Population', 'Muslim Population','Percentage (%) of world Muslim population', 'Sources'],axis=1,inplace=True)

#columns rename
data.columns = ['Country','Islam%']


def precentage(x):
    x = str(x).replace('<','').replace('-','–')
    
    l = [float(i) for i in x.split('–')]
    return np.mean(l)

data['Islam%'] = data['Islam%'].apply(precentage)

# set country as index index
data.set_index('Country',inplace=True)

# copy
IslamData = data.copy()




#       general temeratures data

url =  r'https://en.wikipedia.org/wiki/List_of_countries_by_average_yearly_temperature'
data = pd.read_html(url)[0]

# arrange columns
data.columns = ['Country','avgTemp']

# convert to float
data['avgTemp'] = data['avgTemp'].apply(lambda x:float(x.replace("−", "-")))

# set country as index index
data.set_index('Country',inplace=True)

#copy
tempsData = data.copy()


"""
#        temeratures data of march and april


url =  r'https://en.wikipedia.org/wiki/List_of_cities_by_average_temperature'

for i in range(5):
        
    # read continent temperatures data 
    data = pd.read_html(url)[i]
    
    # select columns batween march and April (can be added if there is a need)
    data = data[['Country','Mar', 'Apr']]
    
    # adajust columns 
    data['Mar'] =  data['Mar'].apply(lambda x: float(x.split('(')[0].replace("−", "-")))
    data['Apr'] =  data['Apr'].apply(lambda x: float(x.split('(')[0].replace("−", "-")))
    data['avgTemp'] = (data['Apr']+data['Mar'])/2
    
    # drop columns 
    data.drop(['Mar','Apr'],axis=1,inplace=True)
    
    # aggregate
    data = data.groupby('Country').mean()
    
    # round digits
    data['avgTemp'] = data['avgTemp'].apply(lambda x :round(x,2))
    
    # if first data - initialize new table
    if i==0 :
        concat = data.copy()
        
    # else - concatinate
    else:
        concat = pd.concat([concat,data])

#copy    
seasonatempData = concat.copy()

"""

#       1st case dates data


# read the 3rd sheet of corona data
path = r'C:\Users\97250\Desktop\studied\R ,python\corona\old_scripts\coronaWorldWide.xlsx'
data = pd.read_excel(path,sheet_name=3)[['Country,Other','Reported1st case']]

# arrange columns
data.columns = ['Country','1stCase']

# set country as index index
data.set_index('Country',inplace=True)

#copy
firstCaseData = data.copy()



#       corona spearded dates data


# shorter version
path = r'C:\Users\97250\Desktop\studied\R ,python\corona\coronaWorldWide_afterTransmission_ver2.xlsx'
x = pd.read_excel(path, sheet_name=None)

# extended version 
path = r'C:\Users\97250\Desktop\studied\R ,python\corona\old_scripts\coronaWorldWide.xlsx'
x = pd.read_excel(path, sheet_name=None)

#   Unified XL sheets only
sheets = [i for i in x]

#  unified columns style
startSheets =  sheets[:4]
lastSheets = sheets[4:]

for i in startSheets:
    x[i].columns= ['Country', 'TotalCases', 'NewCases', 'TotalDeaths', 'NewDeaths',
       'TotalRecovered', 'ActiveCases', 'Critical', 'Cases/1M pop',
       'Deaths/1M pop', '1stcase', 'ratio']   

for i in lastSheets:
    x[i].columns= ['Country', 'TotalCases', 'NewCases', 'TotalDeaths', 'NewDeaths',
       'TotalRecovered', 'ActiveCases', 'Critical', 'Cases/1M pop',
       'Deaths/1M pop', 'TotalTests', 'Tests/1M pop', 'ratio']

#initialize
data = pd.DataFrame(x[sheets[0]]['Country'])

#fill the table with the specified column
for i in sheets:
    current_data = x[i][['Country','Cases/1M pop']] # can choose any filtering column
    data = data.merge(current_data, on='Country')

#set country as index
data.set_index('Country',inplace=True)

#set column names to the reported dates
data.columns = sheets

# create new column of the date that answer the condition in line 4

data['Over100Cases'] = 0
for country in data.index:
    for date in data.columns[:-1]:
        if data.loc[country,date]>=100:   # can choose any threshold
            data.loc[country,'Over100Cases'] = date
            break

data['Over100Cases'] = data['Over100Cases'].apply(lambda x : pd.to_datetime(x) if x!=0 else x)

#copy
countrySpreadData = data.copy()



#       extract coronaVirus spreading time

from DateTime import DateTime

# join datas of 1stCase and spread time
data = countrySpreadData.join(firstCaseData)[['Over100Cases','1stCase']]


# adjust columns
data['1stCase'] = data['1stCase'].apply(lambda time: DateTime(time+', 2020'))
data['Over100Cases'] = data['Over100Cases'].apply(lambda time: DateTime(time) if time!=0 else 0)


#find not NA data
notNAcountries = data[data['Over100Cases']!=0].index
notNAdifferTimeData = data.loc[notNAcountries]

#Add difference in days
notNAdifferTimeData['difference'] = notNAdifferTimeData['Over100Cases']-notNAdifferTimeData['1stCase']
notNAdifferTimeData['difference'] = notNAdifferTimeData['difference'].apply(int)

#days average
avrageDateOver100Cases = round(np.mean(notNAdifferTimeData['difference']),2)

#join datas
data = data.join(notNAdifferTimeData['difference'],how='outer')

#fill NA
data['difference'].fillna(avrageDateOver100Cases,inplace=True)

#copy data of difference
timeSpreadData = data.copy()



#       EU membership


url =  r'https://europa.eu/european-union/about-eu/countries_en#tab-0-0'
data = pd.read_html(url)[0]

#temp column nasmes for concatniation
a = data.columns[0]
b = data.columns[1]

#concat
concat = pd.concat([data[a],data[b]])

#arrange data
concat = concat.iloc[:-1]
concat = concat.reset_index().drop('index',axis=1)

# rename columns
concat.columns = ['Country']

# set country as index index
concat.set_index('Country',inplace=True)

#copy
UEdata = concat.copy()

#timeSpreadData['UE Member'] = timeSpreadData.reset_index()['Country'].apply(lambda x : 1 if x in UEdata else 0)



#intersection method


def intersection(lis):
    
    """
    DESCRIPTION:
        finds intersection between lists of datas
    
    input:
        list of data or serieses
    
    returns:
        Series of intersection
        
    """
    
    if type(lis[0]) == pd.core.frame.DataFrame: # if elements are DF
        for c,i in enumerate(lis):
            if c==0:    
                x = i.index
            else:
                y = i.index
                x = pd.Series(list(set(x) & set(y)))
                return x
    else:                       # if elements are not DF: Serieses...
        for c,i in enumerate(lis):
            if c==0:    
                x = i
            else:
                y = i
                x = pd.Series(list(set(x) & set(y)))
                return x
                    
"""

lets devide into 2 data kinds:
    
Excel - corona, local:
    1. timeSpreadData
    2. coronaData

Wikipedia - 
    1. IslamData
    2. Tempsdata
    3. catholicData
    4. GDPData

   
intersectionOfWikiDatas     = intersection([tempsData, GDPData, catholicData ,IslamData])
intersectionOfCoronaDatas   = intersection([coronaData,timeSpreadData])
intersectionOfAllDatasExceptTimeSpreadData = intersection([tempsData, GDPData, catholicData ,IslamData,coronaData])
intersectionOfAll           = intersection([intersectionOfCoronaDatas,intersectionOfWikiDatas])
    
"""

# join datas method

def join(lis):
    
    """
    joins data by inner join
    """
    
    for c,i in enumerate(lis):
            if c==0:    
                x = i
            else:
                y = i
                x = x.join(y,how='inner')
    return x

# needed intersection        
data = join([tempsData, GDPData, catholicData ,IslamData,coronaData,timeSpreadData])


###         predictive model




data = data.dropna()
X = data.drop(['Deaths/Cases'],axis=1)
y = data['Deaths/Cases']

from sklearn.preprocessing import MinMaxScaler

cols = X.columns
X = MinMaxScaler().fit_transform(X)
X = pd.DataFrame(X,columns = cols)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression().fit(X_train,y_train)


#       evaluation


# coefficients
coefficient = lm.coef_
coefficient_data = pd.DataFrame(coefficient,cols,columns = ['coefficient'])
coefficient_data = coefficient_data.sort_values('coefficient',ascending=False)
print(coefficient_data)

#prediction
prediction = lm.predict(X_test)

# scatterplot of test and predictions
import seaborn as sns
sns.scatterplot(x = y_test,y = prediction)#, hue =(abs(prediction-y_test)))

# plot ditribution of  residuals of the y_test and prediction
residuals = (y_test-prediction)
sns.distplot(residuals,bins = 60,color='red')

# numerical evaluation

MAE = np.mean(abs(prediction-y_test))
MSE = np.mean((prediction-y_test)**2)
RMSE= np.sqrt(np.mean((prediction-y_test)**2))
print('\n')
print('MAE: '+str(MAE),'MSE: '+str(MSE),'RMSE: '+str(RMSE),sep = '\n')




#           ANN




X = X.values
y = y.values


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.advanced_activations import LeakyReLU


#model
model = Sequential()

model.add(Dense(5,activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(2,activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(1))


model.compile(optimizer='adam',loss='mse')
early = EarlyStopping(monitor='val_loss',mode='min',verbose =1,patience=25)
model.fit(X_train,y_train,epochs = 100,validation_data = (X_test,y_test),callbacks =[early])


#visualize

loss = model.history.history['loss']
val_loss = model.history.history['val_loss']
errorDropout = pd.DataFrame(loss,val_loss).reset_index()
errorDropout.columns = ['loss','val_loss']

#plot grath
errorDropout.plot(figsize=(12,12),lw=5,fontsize=24)
plt.legend(loc='upper left',prop={'size':20})

#prediction
pred = model.predict(X_test)
y_test = y_test.reshape(1500,1)


#evaluation

#model.evaluate(X_train,y_train)
MAE =round(mean_absolute_error(pred,y_test))
print('The RMSE is: ' , round(np.sqrt(mean_squared_error(pred,y_test))))
print('The mean absolute error is: ' , round(mean_absolute_error(pred,y_test)))

