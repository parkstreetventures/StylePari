# Load EDA
import numpy as np
import pandas as pd 
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel

# other packages
import nltk
import string
import ast
import re
import unidecode
import unicodedata

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from collections import Counter

import pickle 
import unidecode, ast

import fashion_recommender_color_functions as fc

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# that's all the libaries that we need ..........

# path to all files
DATA_PATH = "./data/saree_data.csv"
CLEAN_PATH = "./data/data_parsed_new.csv"
TFIDF_ENCODING_PATH = "./model/data_tfidf_encodings.pkl"
TFIDF_MODEL_PATH = "./model/data_tfidf.pkl"

#column names
COLUMN_NAME = ['title', 'desc', 'keywords', 'url','score']


def recommendationEngine(color_choice, fabric_choice, N=1):
    color_name = fc.find_name(color_choice)
    #color_name = "red"
    parseDataFile()
    createModel()
    search_term = color_name + " " + fabric_choice
    #getRecommendations(search_term, N=5)
    df1 = getRecommendations(search_term, N)
    df1['color'] = color_name
    complementary = fc.complementaryColor(color_name)
    search_term = complementary + " " + fabric_choice
    df2 = getRecommendations(search_term, N)
    df2['color'] = complementary
    frames = [df1, df2]
    result = pd.concat(frames)
    return result

# Load Our Dataset
def loadData(fileName): 
    df = pd.read_csv(fileName)
    return df 

#save the file
def saveData(df, fileName): 
    df.to_csv(fileName, index=False)  
    

def parseDataFile():
    
    # parses the data into words
    rec_df = loadData(DATA_PATH)
    rec_df[COLUMN_NAME[1]] = rec_df['ingredients']
    
    # change the way the sentence is arranged in the data
    rec_df['ingredients'] = rec_df['ingredients'].map(str) + ',' + rec_df['recipe_name'].map(str)
    rec_df['ingredients'] = rec_df['ingredients'].str.split()

    rec_df['ingredients_parsed'] = rec_df['ingredients'].apply(lambda x: ingredient_parser(x))

    df = rec_df[['recipe_name', 'desc', 'ingredients_parsed', 'ingredients', 'recipe_urls']]
    df = rec_df.dropna()
    # delete the ingredients column
    df.drop(['ingredients'], axis=1,inplace=True)
    # rename all the columns 
    df.rename(columns={'recipe_name': COLUMN_NAME[0], 'ingredients_parsed': COLUMN_NAME[2], 'recipe_urls': COLUMN_NAME[3]}, inplace=True)

    saveData(df,CLEAN_PATH)

def createModel():
   
    # load in parsed recipe dataset     
    df_rec = loadData(CLEAN_PATH)
    df_rec[COLUMN_NAME[2]] = df_rec.desc.values.astype('U')

    # TF-IDF feature extractor 
    tfidf = TfidfVectorizer()
    tfidf.fit(df_rec[COLUMN_NAME[2]])
    tfidf_recipe = tfidf.transform(df_rec[COLUMN_NAME[2]])

    # ------
    #Printing the feature names
    #print(tfidf.get_feature_names())
    #matrix = tfidf_recipe.todense()
    #tfidf_list = matrix.tolist()
    #tfidf_df = pd.DataFrame(tfidf_list, columns = vectorizer.get_feature_names())
    #print(tfidf_df)
    # ------
    
    # save the tfidf model and encodings 
    with open(TFIDF_MODEL_PATH, "wb") as f:
        pickle.dump(tfidf, f)

    with open(TFIDF_ENCODING_PATH, "wb") as f:
        pickle.dump(tfidf_recipe, f)


def getRecommendations(ingredients, N):
    """
    The reccomendation system takes in a list of ingredients and returns a list of top 5 
    recipes based of of cosine similarity. 
    :param ingredients: a list of ingredients
    :param N: the number of reccomendations returned 
    :return: top 5 reccomendations for cooking recipes
    """
    # load in tdidf model and encodings 
    with open(TFIDF_ENCODING_PATH, 'rb') as f:
        tfidf_encodings = pickle.load(f)

    with open(TFIDF_MODEL_PATH, "rb") as f:
        tfidf = pickle.load(f)

    # parse the ingredients using the ingredient_parser 
    try: 
        ingredients_parsed = ingredient_parser(ingredients)
    except:
        ingredients_parsed = ingredient_parser([ingredients])
    
    # use our pretrained tfidf model to encode our input ingredients
    ingredients_tfidf = tfidf.transform([ingredients_parsed])

    # calculate cosine similarity between actual recipe ingreds and test ingreds
    cos_sim = map(lambda x: cosine_similarity(ingredients_tfidf, x), tfidf_encodings)
    scores = list(cos_sim)

    # Filter top N recommendations 
    filtered_recommendations = filterRecommendations(N, scores)
    return filtered_recommendations


# Top-N recomendations order by score
def filterRecommendations(N, scores):
    # load in recipe dataset 
    df_rec = loadData(CLEAN_PATH)
    # order the scores with and filter to get the highest N scores
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]
    # create dataframe to load in recommendations 
    # added "dtype=" to fix a pandas dataframe error
    #recommendation = pd.DataFrame(columns = ['recipe', 'desc', 'ingredients', 'score', 'url'], dtype=object)
    recommendation = pd.DataFrame(columns = [COLUMN_NAME[0], COLUMN_NAME[1], COLUMN_NAME[4], COLUMN_NAME[3]], dtype=object)
    #print (recommendation)
    count = 0
    for i in top:
        #recommendation.at[count, 'url'] = df_rec['recipe_urls'][i]
        recommendation.at[count, COLUMN_NAME[3]] = "./images/download_filename_1.jpg"
        recommendation.at[count, COLUMN_NAME[0]] = title_parser(df_rec[COLUMN_NAME[0]][i])
        recommendation.at[count, COLUMN_NAME[1]] = title_parser(df_rec[COLUMN_NAME[1]][i])
        recommendation.at[count, COLUMN_NAME[2]] = df_rec[COLUMN_NAME[2]][i]
        recommendation.at[count, COLUMN_NAME[4]] = "{:.3f}".format(float(scores[i])) #error here?
        count += 1
    return recommendation


# this is the parser algorithm
#Initialising stopwords for english
stop_words = set(stopwords.words('english'))

# neaten the ingredients being outputted
# this is not used anymore 
def ingredient_parser_final(ingredient):
    
    if isinstance(ingredient, list):
        ingredients = ingredient
    else:
        ingredients = ast.literal_eval(ingredient)
    
    ingredients = ','.join(ingredients)
    ingredients = unidecode.unidecode(ingredients)
    return ingredients

def title_parser(title):
    title = unidecode.unidecode(title)
    return title 

def ingredient_parser(ingreds):
    
    #showStatus("ingredient parser")
    words_to_remove = ['(',')','.','\'','saree', 'matching', 'ba', 'gld', 'without', 'women', 'woman','shubh','self','fresh', 'trendz','oil', 'a', 'and',  'or',  'large', 'extra',  'free', 'small', 'from', 'higher', 'for', 'finely', 'freshly', 'to', 'organic', 'the', 'plain', 'plus' ]
    # The ingredient list is now a string so we need to turn it back into a list. We use ast.literal_eval
    if isinstance(ingreds, list):
        ingredients = ingreds
    else:
        ingredients = ast.literal_eval(ingreds)
    # We first get rid of all the punctuation. We make use of str.maketrans. It takes three input 
    # arguments 'x', 'y', 'z'. 'x' and 'y' must be equal-length strings and characters in 'x'
    # are replaced by characters in 'y'. 'z' is a string (string.punctuation here) where each character
    #  in the string is mapped to None. 
    translator = str.maketrans('', '', string.punctuation)
    lemmatizer = WordNetLemmatizer()
    ingred_list = []
    for i in ingredients:
        i.translate(translator)
        # We split up with hyphens as well as spaces
        items = re.split(' |-', i)
        # Get rid of words containing non alphabet letters
        items = [word for word in items if word.isalpha()]
        # Turn everything to lowercase
        items = [word.lower() for word in items]
      
        # remove stop words
        items = [word for word in items if word not in stop_words]
        
        # remove accents
        items = [unidecode.unidecode(word) for word in items] #''.join((c for c in unicodedata.normalize('NFD', items) if unicodedata.category(c) != 'Mn'))
        items = [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') for word in items]
        
        # Lemmatize words so we can compare words to measuring words
        items = [lemmatizer.lemmatize(word) for word in items]
        # Get rid of common easy words
        items = [word for word in items if word not in words_to_remove]
        # remove all square brackets
        items = [remove_between_square_brackets(word) for word in items]
        # remove all special characters
        items = [remove_special_characters(word) for word in items]
        if items:
            ingred_list.append(' '.join(items)) 
    ingred_list = " ".join(ingred_list)
    return ingred_list

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

