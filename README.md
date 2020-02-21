# DataScience jupyter notebook in Docker

If you want to dive in deeper this are great places to start:
* [https://www.kaggle.com/notebooks](https://www.kaggle.com/notebooks)
* [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)

## Instructions

* make sure you update the path of the volume where your data will be saved in the `docker-compose.yml` file
* run: `docker-compose up`
* from the logs copy the url to localhost with token and paste it in the browser
* stop: `docker-compose down`

## Workshop

### Data exploration

* read the data
```python
filename='top50.csv'
df=pd.read_csv(filename,encoding='ISO-8859-1', index_col=0)
```

* preview the data
* What is the shape of the data? 

* Rename the columns
```python
df.rename(columns={'Track.Name':'track_name','Artist.Name':'artist_name','Beats.Per.Minute':'beats_per_minute','Loudness..dB..':'Loudness(dB)','Valence.':'Valence','Length.':'Length', 'Acousticness..':'Acousticness','Speechiness.':'Speechiness'},inplace=True)
df.head()
```

* Check for null values 
```python
df.isnull().sum()
```

* Fill the  null values
```python
# df.fillna(0)
df.fillna(df.mean(), inplace=True)
df.head()
```

* Get a list of all genres
`genre_list=df['Genre'].values.tolist()`


* list the frequency of all artists
```
popular_artist=df.groupby('artist_name').size()
print(popular_artist)
```

* list all artists
```
artist_list=df['artist_name'].values.tolist()
print(artist_list)
```

* describe the data

* make a nice plot
```
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
newdf = df.select_dtypes(include=numerics)
columns = list(newdf.columns)

for col in columns:
    plt.ylabel('frequency')
    plt.xlabel(col)
    plt.hist(newdf[col], bins=20)
    plt.show()
```

## Training a model

* Extract features and target:
```
x=df.loc[:,['Energy','Danceability','Length','Loudness(dB)','Acousticness']].values
y=df.loc[:,'Popularity'].values
```

* Split in training and testing data
```
# Creating a test and training dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
```

* Make a linear regressor and get the model values
```
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

* analyse the results

```
#Displaying the difference between the actual and the predicted
y_pred = regressor.predict(X_test)
df_output = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df_output)
```
Are these good or bad?

* Quantify this
```
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
```
