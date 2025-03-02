import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import requests
from scipy.sparse import csr_matrix  # Updated import to avoid deprecation warning

from tmdbv3api import TMDb, Movie

# Initialize TMDb API
tmdb = TMDb()
tmdb.api_key = 'YOUR_API_KEY'

# OMDb API key
omdb_api_key = 'YOUR_OMDB_API_KEY'

# Load the NLP model and TF-IDF vectorizer from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl', 'rb'))

def create_sim():
    data = pd.read_csv('main_data.csv')
    print("Data loaded from main_data.csv:")
    print(data.head())  # Debugging print
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    sim = cosine_similarity(count_matrix)
    return data, sim

def rcmd(m):
    m = m.lower()
    try:
        data.head()
        sim.shape
    except:
        data, sim = create_sim()
    if m not in data['movie_title'].unique():
        return 'Sorry! The movie you searched is not in our database. Please check the spelling or try another movie.'
    else:
        i = data.loc[data['movie_title'] == m].index[0]
        lst = list(enumerate(sim[i]))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        lst = lst[1:11]
        return [data['movie_title'][a[0]] for a in lst]

def ListOfGenres(genre_json):
    return ", ".join([genre['name'] for genre in genre_json]) if genre_json else ""

def date_convert(s):
    MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
    y, m, d = s[:4], int(s[5:-3]), s[8:]
    return f"{MONTHS[m-1]} {d}, {y}"

def MinsToHours(duration):
    return f"{duration // 60} hours {duration % 60} minutes" if duration % 60 else f"{duration // 60} hours"

def get_suggestions():
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].str.capitalize())

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    r = rcmd(movie)
    movie = movie.upper()
    
    if isinstance(r, str):  # No such movie found
        return render_template('recommend.html', movie=movie, r=r, t='s')

    tmdb_movie = Movie()
    result = tmdb_movie.search(movie)

    if not result:
        return render_template('recommend.html', movie=movie, r="No movie found on TMDB.", t='s')

    # Get movie details from TMDb
    movie_id = result[0].id
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={tmdb.api_key}')
    data_json = response.json()
    
    imdb_id = data_json.get('imdb_id', '')
    if not imdb_id:
        return render_template('recommend.html', movie=movie, r="No IMDb ID found.", t='s')

    print("IMDB ID:", imdb_id)  # Debugging print

    img_path = f'https://image.tmdb.org/t/p/original{data_json["poster_path"]}'
    genre = ListOfGenres(data_json['genres'])

    # Fetch reviews using OMDb API
    omdb_url = f'http://www.omdbapi.com/?i={imdb_id}&apikey={omdb_api_key}&plot=full'
    omdb_response = requests.get(omdb_url).json()

    if 'Error' in omdb_response:
        return render_template('recommend.html', movie=movie, r="No reviews found on OMDb.", t='s')

    reviews_list = omdb_response.get('Ratings', [])
    reviews_status = ['Good' if float(r['Value'].replace('%', '').replace('/10', '')) > 50 else 'Bad' for r in reviews_list]

    # Combine reviews and comments into a dictionary
    movie_reviews = {reviews_list[i]['Source']: reviews_status[i] for i in range(len(reviews_list))}

    vote_count = "{:,}".format(result[0].vote_count)
    rd = date_convert(result[0].release_date)
    status = data_json['status']
    runtime = MinsToHours(data_json['runtime'])

    # Fetch posters for recommended movies
    movie_cards = {}
    for movie_title in r:
        list_result = tmdb_movie.search(movie_title)
        if list_result:
            movie_id = list_result[0].id
            response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={tmdb.api_key}')
            data_json = response.json()
            movie_cards[f'https://image.tmdb.org/t/p/original{data_json["poster_path"]}'] = movie_title

    return render_template(
        'recommend.html',
        movie=movie,
        mtitle=r,
        t='l',
        cards=movie_cards,
        result=result[0],
        reviews=movie_reviews,
        img_path=img_path,
        genres=genre,
        vote_count=vote_count,
        release_date=rd,
        status=status,
        runtime=runtime
    )

if __name__ == '__main__':
    app.run(debug=True)
