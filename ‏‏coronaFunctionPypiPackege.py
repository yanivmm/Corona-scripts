# imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# import EXCEL file with all sheets
path = r'coronaWorld.xlsx'
file = pd.read_excel(path, sheet_name=None)

def plotGraph(column ,numFilter=100 ,top=(0,10) ,add=['Israel'] , drop=[] ,dates=0 ,plot='line'):


    """
    Plots corona virus graphs.


    Parameters
    ----------

    column : string ,required.
        DESCRIPTION: Column name to show data about.
        NO DEAFAULT VALUE.
        The options are: ['Total Cases', 'New Cases', 'Total Deaths','New Deaths',
                        'Total Recovered', 'Active Cases', 'Critical', 'Cases/1M pop',
                        'Deaths/1M pop', 'Total Tests', 'Tests/1M pop','Deaths/Cases',
                        'Tests/Cases','Recovered/Cases','Tests/Cases/1M pop'].

    numFilter : int ,optional.
            DESCRIPTION: Num of 'Total Cases' filter for countries to show.
            DEFAULT: 100.

    top : touple , optional.
            DESCRIPTION: Top range countries to show, ordered by 'column'.
            To show the buttom x countries : top=(-x,-1)
            DEFAULT: (0,10).  (top 10 countries)

    add : list , optional.
            DESCRIPTION: Add countries.
            DEFAULT: ['Israel'] (The best country ever!).
            NOTICE: If Israel is already at the top x countries, you may get an error.
                    In this case ,specify: add=[]

    drop : list , optional.
            DESCRIPTION: Countries you wish to drop.
            DEFAULT: None.
            NOTE: you must enter countries that are already in 'top-countries'.

    dates : int ,optional.
            DESCRIPTION: Number of dates fron current day backwords.
            DEFAULT: 0 - All Dates Available.
            No error expected when given larger number than available dates.

    plot : string , optional.
            DESCRIPTION: plot kind.
            DEFAULT: 'line'.
            You can choose: ['line','bar','agg'].
            Line will show data through time .
            Bar  will show data of today.
            Agg  will show data of today for whole world.


    Returns
    -------
    TYPE: None type.
        DESCRIPTION: plots graph.

    """

    # style
    sns.set_style('darkgrid')

    # define XL sheets by dates
    sheets = [i for i in file][-dates:]


    # filter by over 100 cases

    for i in sheets:
        file[i]=file[i][file[i]['Total Cases']>numFilter]


    ###         creats Time Series table of the column


    #initialize
    data = pd.DataFrame(file['2020-04-20']['Country'])

    #fill the table with the specified column
    for i in sheets:
        current_data = file[i][['Country',column]]
        data = data.merge(current_data, on='Country')

    #set country as index
    data.set_index('Country',inplace=True)

    #set column names to the reported dates
    data.columns = sheets


    #          collect countries data


    # Sorting

    if plot =='line':   # Sort by the Avg.
        top_countries = data.transpose().mean().sort_values(ascending=False)
    else :              # Srot by the current date
        top_countries = data.transpose().iloc[-1].sort_values(ascending=False)


    # grab top countries indexes
    top_countries = top_countries.iloc[top[0]:top[1]].index
    # add/drop
    top_countries = top_countries.append(pd.Index(add)).drop(drop)
    # grab data by dates
    selected_countries_data = data.transpose().loc[sheets]
    # grab data by selected countries
    selected_countries_data = selected_countries_data[top_countries]


    # plot graphs throuth time

    if  plot == 'line':
        title = "\n"+ column + " between "+sheets[0]+ " and " + sheets[-1]+ "\n"
        selected_countries_data.plot(lw = 5,figsize=(12,10), fontsize=18)
        plt.legend(loc='upper left',prop={'size':16})
        plt.title(title,fontsize=30)
        plt.locator_params(nbins=11)

    # plot bar of specific day

    elif plot == 'bar':
        title = "\n " + column + " as of the " + sheets[-1] + "\n"
        selected_countries_data.iloc[-1].plot(kind=plot,figsize=(12,8),color='gold',fontsize = 18)
        plt.title(title,fontsize=34)
        plt.xlabel('')

    # plot one summering data for all countries

    elif plot == 'agg':

        # If columnm is a ratio kind column,
        # it's not wise to sum ratios as total world ratio.
        # we may get 200% deaths/cases.

        if column in ['Cases/1M pop','Deaths/1M pop','Tests/1M pop','Deaths/Cases',
                        'Tests/Cases','Recovered/Cases','Tests/Cases/1M pop']    :
            print('\t it is not wise to aggregate a ratio column .\n\t The avg. of a raio column is not the weighted ratio .\n\t Please try a \'Total kind\' column.')

        else:
            totalAmount = round(data.transpose().iloc[-1].sum())
            print('The ' + column +' of the world as of the '+sheets[-1] +' is: \n\n\t\t\t ' ,totalAmount)
