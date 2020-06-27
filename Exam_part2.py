#!/usr/bin/env python
# coding: utf-8

# # Peer-graded Assignment: Segmenting and Clustering Neighborhoods in Toronto
# 
# 

# ## Part 1

# #### First we import the relevant libraries 

# In[1]:


import pandas as pd # library for data analsysis
import requests # library to handle requests
import urllib.request
import json # library to handle JSON files
get_ipython().system('pip install BeautifulSoup4')
get_ipython().system('pip install html5lib')
get_ipython().system('pip install lxml')
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe
from bs4 import BeautifulSoup


# #### We download the wikipedia page using the BeutifulSoup library 

# In[2]:


url = "https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M"
page = urllib.request.urlopen(url)
soup = BeautifulSoup(page, "html5lib")


# #### We print the wikipedia page and identify the relevant postal code table 

# In[3]:


print(soup.prettify())


# #### Now we isolate reading of the postal code table in the Wikipedia article 

# In[4]:


postal_code_table = soup.find('table', class_='wikitable sortable')


# #### We download the information from the table and insert them into a table with three collumns

# In[5]:


A = []
B = []
C = []

for row in postal_code_table.findAll('tr'):
    cells=row.findAll('td')
    if len(cells)==3:
        A.append(cells[0].find(text=True))
        B.append(cells[1].find(text=True))
        C.append(cells[2].find(text=True))


# In[6]:


df_postal_code = pd.DataFrame(A,columns=['Postal code'])
df_postal_code['Borough'] = B
df_postal_code['Neighborhood'] = C
df_postal_code


# #### The imported table has newlines (\n) included in the table cells. These will be removed

# In[7]:


df_postal_code = df_postal_code.replace("\n", "", regex = True)
df_postal_code


# #### The Boroughs with a Not assigned cell should be deleted. We identify all the relevant rows and delete them 

# In[8]:


indexNames = df_postal_code[ df_postal_code['Borough'] == "Not assigned" ].index
df_postal_code.drop(indexNames , inplace=True)
df_postal_code


# #### We group neighborhoods with the same postal code togehter 

# In[9]:


df_postal_code.rename({"Postal code": "PostalCode"}, axis = 1, inplace = True)
df_postal_code.groupby('PostalCode')['Neighborhood'].agg(','.join)
df_postal_code.reset_index(inplace = True, drop=True)


# In[10]:


df_postal_code.shape


# ## Part 2

# #### As geopy failed to look up Neigborhoods with a "-", we read the csv. file using the URL

# In[11]:


X = pd.read_csv("http://cocl.us/Geospatial_data")


# #### We take a look at the structure of the CSV-file 

# In[12]:


print(X)


# #### We rename the indicator PostalCode to merge with the postal code data frame

# In[13]:


X.rename({"Postal Code": "PostalCode"}, axis = 1, inplace = True)


# #### We merge our dataframe with the CSV-file using PostalCode as common reference

# In[14]:


postal_code_merged = pd.merge(df_postal_code, X, on="PostalCode")
postal_code_merged.head()


# ## Part 3

# #### To analyze the neigborhoods We first import the relevant libraries 

# In[15]:


# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes')
import folium # map rendering library

get_ipython().system("conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab")
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import numpy as np

print('Libraries imported.')


# #### We isolate boroughs containing the word "Toronto"

# In[16]:


df_Toronto = postal_code_merged[postal_code_merged["Borough"].str.contains("Toronto")]
df_Toronto.reset_index(inplace = True, drop=True)
print(df_Toronto.shape)


# In[17]:


df_Toronto


# #### We define the Foursqaure credentials and version

# In[18]:


CLIENT_ID = 'SU2HMHHI1250IJUBBGDAVXC1XH5PDACNNDOGRRIGXXDGYB0P' # Foursquare ID
CLIENT_SECRET = 'VE3NZTFTTPY3PLDLJW5G0H0GUUAHA1MSKWNMNCJOCSJEQAUS' # Foursquare Secret
VERSION = '20180605' # Foursquare API version


# #### We define the Get requests

# In[19]:


address = 'Davisville, Toronto, Canada'

geolocator = Nominatim(user_agent="foursquare_agent")
location = geolocator.geocode(address)
latitude =  43.654260
longitude = -79.360636 
radius = 500
LIMIT = 100

url = 'https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude, VERSION, radius, LIMIT)
url


# In[20]:


results = requests.get(url).json()


# #### Define a function that extracts the category of the venue

# In[21]:


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[22]:


venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]
# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()


# In[23]:


print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


# In[24]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# #### Now write the code to run the above function on each neighborhood and create a new dataframe called Toronto_venues

# In[25]:


Toronto_venues = getNearbyVenues(names=df_Toronto['Neighborhood'],
                                   latitudes=df_Toronto['Latitude'],
                                   longitudes=df_Toronto['Longitude'])


# In[26]:


print(Toronto_venues.shape)
Toronto_venues.head()


# In[27]:


Toronto_venues.groupby('Neighborhood').count()


# #### Analyze each neigborhood 

# In[28]:


# one hot encoding
Toronto_onehot = pd.get_dummies(Toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
Toronto_onehot['Neighborhood'] = Toronto_venues['Neighborhood'] 

# move neighborhood column to the first column
index = Toronto_onehot.columns.get_loc("Neighborhood")
fixed_columns = [Toronto_onehot.columns[index]] + list(Toronto_onehot.columns[:index]) + list(Toronto_onehot.columns[index+1:])
Toronto_onehot = Toronto_onehot[fixed_columns]

Toronto_onehot.head()


# In[29]:


Toronto_grouped = Toronto_onehot.groupby('Neighborhood').mean().reset_index()
Toronto_grouped.head()


# In[30]:


Toronto_grouped.shape


# #### We write a function to sort the venues in descending order.

# In[31]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[32]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = Toronto_grouped['Neighborhood']

for ind in np.arange(Toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(Toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# #### We cluster the neigborhoods 

# In[33]:


# set number of clusters
kclusters = 5

Toronto_grouped_clustering = Toronto_grouped.drop('Neighborhood', axis=1)
# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(Toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:38] 


# In[34]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

Toronto_merged = df_Toronto

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
Toronto_merged = Toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

Toronto_merged.head() # check the last columns!


# #### We find the geographical coordinates of Toronto to use in for mapping the neighborhoods

# In[35]:


address = 'Toronto, Canada'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinates of Toronto are {}, {}.'.format(latitude, longitude))


# #### We visualize the resulting cluster value for each neigborhoood

# In[36]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(Toronto_merged['Latitude'], Toronto_merged['Longitude'], Toronto_merged['Neighborhood'], Toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[ ]:




