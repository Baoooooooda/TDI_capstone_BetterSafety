import pandas as pd
import os
from sodapy import Socrata
import requests
import dill
import numpy as np
from math import pi
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium import plugins
import logging
import argparse
from datetime import datetime
import pmdarima as pm


log_format = '%(asctime)s - %(levelname)s: %(message)s'
formatter = logging.Formatter(log_format)

def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file, encoding = "utf-8")
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

try:
    if not os.path.isdir("log"):
        os.makedirs("log")

    log_name = "log/" + datetime.now().strftime("%Y%m%d") + "_handler.log"

    lg = setup_logger('main', log_name)
except IndexError:
    pass

def input_parser():
    parser = argparse.ArgumentParser(description='Enter the address and html page.')
    parser.add_argument('-a', '--address', type=str, required=True, help='Enter the address')
    parser.add_argument('-p', '--page', type=str, required=True, help='Which html page')
    args = parser.parse_args()
    return vars(args)


#0.##########################################################################################
# Helper function to preprocess the data
# Load dataframe saved

#pkd = "seattle_gov_df.pkd"

def load_df(pkd):
    with open(pkd, 'rb') as f:
        df = dill.load(f)
    return df

# The date will be the start of lifting stay-at-home probabily
# def get_n_days_data(x, today = pd.to_datetime('today'), n = 544, date = '2021-6-30'):
#    if n:
#        mask1 = (x.offense_start_datetime >= pd.to_datetime(-n, unit = 'D', origin = today))\
#        & (x.offense_start_datetime < today)
#    else:
#        mask1 = x.offense_start_datetime > date
#    return x.loc[mask1]

# Get only Group A and Group B '90J' data
# def filter_group(x):
#    mask = (x.group_a_b == 'A')\
#    | ((x.group_a_b == 'B') & (x.offense_code == '90J')) # only take in Group A and Group B 90J
#    return x.loc[mask]

# Get the geocode of the user input address
def get_geocode(address):
    params = { 'format'        :'json',
               'addressdetails': 1,
               'q'             : address}
    headers = { 'user-agent'   : 'YRW' }   #  Need to supply a user agent other than the default provided
                                           #  by requests for the API to accept the query.
    response = requests.get('http://nominatim.openstreetmap.org/search',
                        params=params, headers=headers)
    lat = response.json()[0]['lat']
    lon = response.json()[0]['lon']
    lg.info(address)
    lg.info(lat)
    return float(lat), float(lon)


# Filter out invalid geographical information
# def valid_location(x):
#    mask = ~((x.longitude == 0) | (x.latitude == 0))
#    return x.loc[mask]

# Calculate distance of two points by geocode
def pipe_distance(x, latA, lonA):
#    latA, lonA = map(float, get_geocode(address))
    distance = np.sin(np.radians(latA)) * np.sin(np.radians(x['latitude'])) + np.cos(np.radians(latA)) * np.cos(np.radians(x['latitude'])) \
* np.cos(np.radians(lonA - x['longitude']))
    distance_mile = np.arccos(distance) * 180 / pi * 69.09
    x['distance'] = distance_mile
    return x

# Check select the data within N mile
def pipe_distance_checker(x, mile = 3):
    if mile:
        mask = x.distance <=mile
        return x.loc[mask]
    else:
        return x

# Main function to get the dataframe with selected condition
def get_nmile(df1, lat, lon,  mile = 3):
    df = df1.pipe(pipe_distance, lat, lon).pipe(pipe_distance_checker, mile = mile)
    return df
####################################################################



# Get the dataframe for map, local/city compare, dayofweek and crimebyhour figure

#1.#####################################################################
## Density map "crimemap.html"
def density_map(map_df, lat, lon):
    m1 = folium.Map([lat, lon], zoom_start = 13)
    for index, row in map_df.iterrows():
        folium.CircleMarker([row['latitude'], row['longitude']],
                    radius=1,
                    popup=row['offense_id'],
                    fill_color="#3db7e4"
                   ).add_to(m1)

    folium.Marker([lat, lon], icon = folium.Icon(color = 'red')).add_to(m1)
    dfmatrix = map_df[['latitude', 'longitude']].values
    m1.add_child(plugins.HeatMap(dfmatrix, radius=15))
    m1.save('static/crimemap.html')
    return None
######################################################################

#2.#####################################################################
## Get percentage bar plot "crime_count.html"
def percentage_plot(df1, dfmap):
    percent_df = percentage(df1, dfmap)
    crime = alt.selection_single()
    # one_mile = alt.Chart(crime_plot.reset_index()).mark_bar().encode(x =alt.X('one_mile:Q', axis = alt.Axis(title = 'Total crime number (within one mile)')),
    #                                                                y = alt.Y('index:O', axis = alt.Axis(title = 'Offense type')),
    #                                                                 color = alt.condition(crime, alt.value('blue'), alt.value('lightgray')))\
    #                                                                .properties(height = 700).add_selection(crime)
    one_mile = alt.Chart(percent_df).mark_bar().encode(x =alt.X('local:Q', axis = alt.Axis(title = f'Total crime number (within 3 miles)', titleFontSize=20, labelFontSize = 20)),
                                                                   y = alt.Y('offense:O', sort = "-x", axis = alt.Axis(title = 'Offense type', titlePadding = 140, titleFontSize = 22, labelFontSize = 16, labelPadding = 5, labelLimit = 2300)))
    text = one_mile.mark_text(
        align='left',
        baseline='middle',
        dx=3,  # Nudges text to right so it doesn't appear on top of the bar
        fontSize = 16,
        fontWeight = "bold"
).encode(
        text='percentage:O'
    )

    combined = (one_mile + text).properties(height=500, width= 400)
    combined = combined.encode(color = alt.condition(crime, alt.value('blue'), alt.value('lightgray'))).add_selection(crime)

    citywide = alt.Chart(percent_df).mark_bar().encode(x =alt.X('citytotal:Q', axis = alt.Axis(title = 'Total crime number (citywide)', titleFontSize = 20, labelFontSize = 20)),
                                                                   y = alt.Y('offense:O', sort = "-x", axis = alt.Axis(title = (''), labels = True, labelFontSize = 16)),
                                                                    color = alt.condition(crime, alt.value('blue'), alt.value('lightgray')))\
                                                                   .properties(height = 500, width=400).add_selection(crime)
    chart = (combined | citywide)
    chart.save('static/crime_count.html')
    return chart

# Helper function to preprocess the data
def percentage(df1, dfmap):
    city_df = df1.set_index('offense').groupby('offense')[['offense_id']].count()
    local_df = dfmap.set_index('offense').groupby('offense')[['offense_id']].count()
    merged_df = city_df.merge(local_df, left_index = True, right_index = True, how = 'left')
    merged_df = merged_df.rename(columns = {'offense_id_x' : 'citytotal', 'offense_id_y' : 'local'})
    merged_df['local'].fillna(0, inplace = True)
    merged_df['percentage'] = round(merged_df.local/merged_df.citytotal*100, 1)
    merged_df['percentage'] = merged_df['percentage'].apply(lambda x: f"{x}%")
    merged_df.reset_index(inplace = True)
    merged_df = merged_df.sort_values('local', ascending = False)
    return merged_df


#3.###############################################################################
# Plot the dayofweek figure "dayofweek.jpg"
def plot_day(dfmap):
    df_plot = dfmap.pipe(get_merged_real_ref).reset_index('offense').reset_index('dayofweek')
    sns.set(font_scale = 1.5)
    g = sns.FacetGrid(data = df_plot, row = 'offense', height = 4, aspect = 2)
    g.map_dataframe(sns.barplot, x = 'dayofweek', y = 'offense_id', color = 'blue')
    g.figure.subplots_adjust(hspace = 0.2)
    g.axes[0, 0].set_ylabel('Number of Crimes')
    g.axes[1, 0].set_ylabel('Number of Crimes')
    g.axes[2, 0].set_ylabel('Number of Crimes')
    g.axes[3, 0].set_ylabel('Number of Crimes')
    g.axes[4, 0].set_ylabel('Number of Crimes')
    g.axes[4, 0].set_xticklabels(g.axes[4, 0].get_xticklabels(), rotation = 45)
    g.axes[4, 0].set_xlabel('Days of Week', fontsize = 20)
    return g.savefig('static/dayofweek.jpg')

# Helper function
def get_merged_real_ref(df):
    real = df.pipe(grouped_top5)
    ref = df.pipe(get_ref_df)
    merged = ref.merge(real, left_index = True, right_index = True, how = 'left')
    merged['offense_id'] = merged.offense_id.fillna(0)
    merged.drop(columns = 0, axis = 1, inplace = True)
    return merged

def grouped_top5(df):
    grouped = topfivecrimedf(df)\
              .pipe(add_day_hour)\
              .pipe(grouped_df)
    return grouped

def topfivecrimedf(df):
    mask = topfivecrime(df)
    df = df.loc[df.offense.isin(mask) == True]
    return df

def topfivecrime(df):
    top5 = df.groupby('offense')['offense_id'].count().sort_values(ascending = False).head().index
    return top5

def add_day_hour(df):
    df['dayofweek'] = df.offense_start_datetime.dt.day_name()
    df['hourofday'] = df.offense_start_datetime.dt.hour
    return df

def grouped_df(df):
    grouped = df.groupby(['offense', 'dayofweek'])[['offense_id']].count()
    return grouped

def get_ref_df(df):
    index_tuple = crime_day_pair(df)
    multiindex = pd.MultiIndex.from_tuples(index_tuple, names = ('offense', 'dayofweek'))
    ref_df = pd.DataFrame(np.zeros(len(multiindex)), index = multiindex)
    return ref_df

def crime_day_pair(df):
    daylist = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pairs = []
    for crime in topfivecrime(df):
        for day in daylist:
            pair = (crime, day)
            pairs.append(pair)
    return pairs

################################################################################


#4.###############################################################################
# Plot the crimes by hour "crimebyhour.html"
def hourplot(dfmap):
    # Make proper dataframe
    hour = dfmap.pipe(add_day_hour)
    hour = hour.groupby('hourofday')[['offense_id']].count()/7
    hour = hour.rename(columns = {'offense_id' : 'Number of Crimes'}).reset_index()
    hour_am = hour[hour.hourofday <=11]
    hour_pm = hour_pm = hour[hour.hourofday >11]
    import plotly.graph_objects as go

    # Plot and save the figure
    from plotly.subplots import make_subplots
    #scope = PlotlyScope(
    #    plotlyjs="https://cdn.plot.ly/plotly-latest.min.js",
    #    # plotlyjs="/path/to/local/plotly.js",
    #)


    fig = make_subplots(rows = 1, cols = 2, specs = [[{'type': 'polar'}]*2],
                        subplot_titles = ('Crime Count from 0AM - 12AM', 'Crime Count from 12PM - 0AM'))
    fig.add_trace(go.Barpolar(
        r = hour_am['Number of Crimes'],
        theta = ['0am', '1am', '2am', '3am', '4am', '5am', '6am', '7am', '8am', '9am', '10am', '11am'],
    #     theta=(hour_am['hourofday']+0.5)*30,
        marker_color= 'blue',
        marker_line_color="black",
        marker_line_width=1,
        opacity=0.8
    ), 1, 1)

    fig.add_trace(go.Barpolar(
        r = hour_pm['Number of Crimes'],
        theta = ['12pm', '1pm', '2pm', '3pm', '4pm', '5pm', '6pm', '7pm', '8pm', '9pm', '10pm', '11pm'],
    #     theta=(hour_am['hourofday']+0.5)*30,
        marker_color= 'orange',
        marker_line_color="black",
        marker_line_width=1,
        opacity=0.8
    ), 1, 2)

    fig.update_layout(
        template=None,
        polar = dict(
            radialaxis = dict(range=[0, max(max(hour_am['Number of Crimes']), max(hour_pm['Number of Crimes']))], showticklabels=True, ticks=''),
            angularaxis = dict(showticklabels=True, ticks='',
    #                            categoryarray = ['0am', '1am', '2am', '3am', '4am', '5am', '6am', '7am', '8am', '9am', '10am', '11am'],
    #                            categoryorder = "array",
                               direction = 'clockwise', showgrid = False,
                              linewidth = 2)

        ),
        polar2 = dict(
            radialaxis = dict(range=[0, max(max(hour_am['Number of Crimes']), max(hour_pm['Number of Crimes']))], showticklabels=True, ticks=''),
            angularaxis = dict(showticklabels=True, ticks='',
    #                            categoryarray = ['0am', '1am', '2am', '3am', '4am', '5am', '6am', '7am', '8am', '9am', '10am', '11am'],
    #                            categoryorder = "array",
                               direction = 'clockwise', showgrid = False,
                              linewidth = 2)

        )
    )

    fig.update_layout(font = dict(size = 20), height = 660, width = 1060, title = 'Crime Count by Hours', showlegend = False, title_font_size = 25)
#     with open("crimebyhour1.jpg", "wb") as f:
#         f.write(scope.transform(fig, format = "jpg"))
    fig.write_html('static/crimebyhour.html')
    return None


##############################################################
# Get the predicted days and week figures
def prepare_df(df):
    df = df.loc[df['crime_against_category'] == 'PERSON']
    df = df[['offense_start_datetime', 'offense_id']]
    df = df.set_index('offense_start_datetime').resample('D')[['offense_id']].count()
    df = df[:-1]
    return df


def prediction_day(df):
    df = prepare_df(df)
    model_auto = pm.auto_arima(df['offense_id'],start_p=0,d=1,start_q=0,
          max_p=5,max_d=5,max_q=5, start_P=0,
          D=1, start_Q=0, max_P=5,max_D=5,
          max_Q=5, m=7, seasonal=True,
          error_action='warn',trace=True,
          supress_warnings=True,stepwise=True,
          random_state=20,n_fits=30) 
    predictions = model_auto.predict(n_periods=21)
    date_index = pd.date_range(start = pd.to_datetime('today'), end = pd.to_datetime(21, unit = 'D', origin = pd.to_datetime('today')))
    df_future = pd.DataFrame(model_auto.predict(n_periods = 21),index=date_index)
    fig1, ax1 = plt.subplots(figsize = (7, 4))
    ax1.plot(df_future.index, df_future.values, '-bo', label = False)
    plt.tight_layout()
    plt.xticks(rotation = 45, fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.xlabel('Future days', fontsize = 20)
    plt.ylabel('Crime count', fontsize = 20)
    plt.title('Predicted crime count', fontsize = 22) 
    
    # Get the weekly plot
    df_future['dayofweek'] = df_future.index.day_name()
    sorter = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_future['dayofweek'] = df_future['dayofweek'].astype('category')
    df_future['dayofweek'] = df_future['dayofweek'].cat.set_categories(sorter)
    df_week = df_future.groupby('dayofweek')[0].mean()
    fig2, ax2 = plt.subplots(figsize = (7, 4))
    ax2.plot(df_week.index, df_week.values, "-bo")
    plt.xticks(rotation = 45, fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.xlabel('Days of week', fontsize = 20)
    plt.ylabel('Crime count', fontsize = 20)
    plt.title('Predicted weekly crime trend', fontsize = 22) 
    
    # Save figures
    fig1.savefig('static/prediction_day.jpg', bbox_inches = 'tight')
    fig2.savefig('static/prediction_week.jpg', bbox_inches = 'tight')
    return None

# Get the model figure
def show_model(df):
    df = prepare_df(df)
    size = 21
    train = df[:-21]
    test = df[-21:]
    model_auto = pm.auto_arima(train['offense_id'],start_p=0,d=1,start_q=0,
          max_p=5,max_d=5,max_q=5, start_P=0,
          D=1, start_Q=0, max_P=5,max_D=5,
          max_Q=5, m=7, seasonal=True,
          error_action='warn',trace=True,
          supress_warnings=True,stepwise=True,
          random_state=20,n_fits=30)  
    in_sample = model_auto.predict_in_sample()
    test_predictions = model_auto.predict(n_periods=21)
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.plot(df.index, df['offense_id'], label='True crime count', color='b')
    ax.plot(df.index[:-21], in_sample, label='In-sample predictions', color='orange')
    ax.plot(df.index[-21:], test_predictions, label='Test set predictions', color='r')
    ax.legend(loc="lower right", fontsize = 14)
    plt.tight_layout()
    plt.xticks(rotation = 45, fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.xlabel('Past days', fontsize = 20)
    plt.ylabel('Crime count', fontsize = 20)
    plt.title('SARIMA model', fontsize = 22)
    fig.savefig('static/model.jpg', bbox_inches = 'tight')
    return None


if __name__ == "__main__":
   
    config = input_parser() 
    address = config["address"]
    page = config["page"]

    reallat, reallon = get_geocode(address)

    if page == "1":
        pkd1 = "./preprocess_crime.pkd"
        df1 = load_df(pkd1)
        dfmap = get_nmile(df1, reallat, reallon, mile = 3)
        density_map(dfmap, reallat, reallon)
        percentage_plot(df1, dfmap)
        plot_day(dfmap)
        hourplot(dfmap)
    else:
        pkd2 = "./preprocess_time.pkd"
        df2 = load_df(pkd2)
        dftime = get_nmile(df2, reallat, reallon, mile = 3)
        prediction_day(dftime) #generate prediction_day.jpg and prediction_week.jpg
        show_model(dftime) #generate model.jpg
    


     
