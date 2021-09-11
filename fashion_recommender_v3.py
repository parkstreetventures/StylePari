# this is the latest update to the recommender
# we will save the model and reuse instead of creating it all the time
# v3 Sep 2021

# run this program - streamlit run /home/ec2-user/project-python/StylePari/fashion_recommender_v3.py
#  
# Core Pkg
import streamlit as st 
import streamlit.components.v1 as stc 


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

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter

import pickle 
import unidecode, ast


#nltk.download('wordnet')

# that's all the libaries that we need ..........

# path to all files
RECIPES_PATH = "./data/saree_data.csv"
PARSED_PATH = "./data/saree_parsed_new.csv"
TFIDF_ENCODING_PATH = "./model/saree_tfidf_encodings.pkl"
TFIDF_MODEL_PATH = "./model/saree_tfidf.pkl"

# color functionality 

import re
re_color = re.compile('#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})')
from math import sqrt

def color_to_rgb(color):
    return tuple(int(x, 16) / 255.0 for x in re_color.match(color).groups())

def similarity(color1, color2):
    """Computes the pearson correlation coefficient for two colors. The result
    will be between 1.0 (very similar) and -1.0 (no similarity)."""
    c1 = color_to_rgb(color1)
    c2 = color_to_rgb(color2)

    s1 = sum(c1)
    s2 = sum(c2)
    sp1 = sum(map(lambda c: pow(c, 2), c1))
    sp2 = sum(map(lambda c: pow(c, 2), c2))
    sp = sum(map(lambda x: x[0] * x[1], zip(c1, c2)))

    try:
            computed = (sp - (s1 * s2 / 3.0)) / sqrt((sp1 - pow(s1, 2) / 3.0) * (sp2 - pow(s2, 2) / 3.0))
    except:
            computed = 0
    
    return computed

color_names = {
    '#000000': 'black',
    '#ffffff': 'white',
    '#808080': 'dark gray',
    '#b0b0b0': 'light gray',
    '#ff0000': 'red',
    '#800000': 'dark red',
    '#00ff00': 'green',
    '#008000': 'dark green',
    '#0000ff': 'blue',
    '#000080': 'dark blue',
    '#ffff00': 'yellow',
    '#808000': 'olive',
    '#00ffff': 'cyan',
    '#ff00ff': 'magenta',
    '#800080': 'purple'
    }

def find_name(color):
    sim = [(similarity(color, c), name) for c, name in color_names.items()]
    return max(sim, key=lambda x: x[0])[1]


# end color functionality

# debugging help file
def showStatus(moduleName):
    debug_mode="n"
    if debug_mode=="Y":
        print(moduleName)

# to display images within Jupyter ..
def path_to_image_html(path):
    return '<img src="'+ path + '" width="80" >'

# Load Our Dataset
def load_data(fileName): 
	df = pd.read_csv(fileName)
	return df 

#save the file
def save_data(df, fileName): 
    df.to_csv(fileName, index=False)  

# recommendation system

# Top-N recomendations order by score
def get_recommendations(N, scores):
    # load in recipe dataset 
    df_recipes = load_data(PARSED_PATH)
    # order the scores with and filter to get the highest N scores
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]
    # create dataframe to load in recommendations 
    # added "dtype=" to fix a pandas dataframe error
    #recommendation = pd.DataFrame(columns = ['recipe', 'desc', 'ingredients', 'score', 'url'], dtype=object)
    recommendation = pd.DataFrame(columns = ['recipe', 'desc', 'score', 'url'], dtype=object)
    #print (recommendation)
    count = 0
    for i in top:
        #recommendation.at[count, 'url'] = df_recipes['recipe_urls'][i]
        recommendation.at[count, 'url'] = "./images/download_filename_1.jpg"
        recommendation.at[count, 'recipe'] = title_parser(df_recipes['recipe_name'][i])
        recommendation.at[count, 'desc'] = title_parser(df_recipes['desc'][i])
        #recommendation.at[count, 'ingredients'] = ingredient_parser_final(df_recipes['ingredients'][i])
        recommendation.at[count, 'score'] = "{:.3f}".format(float(scores[i])) #error here?
        count += 1
    return recommendation

# neaten the ingredients being outputted
# this is not used anymore 
def ingredient_parser_final(ingredient):
    showStatus("ingredient parser final")
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

def RecSys(ingredients, N=5):
    """
    The reccomendation system takes in a list of ingredients and returns a list of top 5 
    recipes based of of cosine similarity. 
    :param ingredients: a list of ingredients
    :param N: the number of reccomendations returned 
    :return: top 5 reccomendations for cooking recipes
    """
    showStatus("recommendation system")
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
    #print(scores)

    # Filter top N recommendations 
    recommendations = get_recommendations(N, scores)
    return recommendations


def ingredient_parser(ingreds):
    '''
    
    This function takes in a list (but it is a string as it comes from pandas dataframe) of 
       ingredients and performs some preprocessing. 
       For example:
       input = '['1 x 1.6kg whole duck', '2 heaped teaspoons Chinese five-spice powder', '1 clementine',
                 '6 fresh bay leaves', 'GRAVY', '', '1 bulb of garlic', '2 carrots', '2 red onions', 
                 '3 tablespoons plain flour', '100 ml Marsala', '1 litre organic chicken stock']'
       
       output = ['duck', 'chinese five spice powder', 'clementine', 'fresh bay leaf', 'gravy', 'garlic',
                 'carrot', 'red onion', 'plain flour', 'marsala', 'organic chicken stock']
    '''
    #showStatus("ingredient parser")
    measures = [ 'cup', 'c', 'p', 'pt',  'deciliter', 'decilitre',  'pound', 'lb', '#', 'ounce', 'oz', 'mg', 'milligram', 'milligramme', 'g', 'gram', 'gramme', 'kg', 'kilogram', 'kilogramme', 'x', 'of', 'mm', 'millimetre', 'millimeter', 'cm', 'centimeter', 'centimetre', 'm', 'meter', 'metre', 'inch', 'in', 'milli', 'centi', 'deci', 'hecto', 'kilo']
    words_to_remove = ['(',')','.','\'','with', 'matching', 'ba', 'gld', 'without', 'women','fresh', 'oil', 'a', 'and',  'or',  'large', 'extra',  'free', 'small', 'from', 'higher', 'for', 'finely', 'freshly', 'to', 'organic', 'the', 'plain', 'plus' ]
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
        # remove accents
        items = [unidecode.unidecode(word) for word in items] #''.join((c for c in unicodedata.normalize('NFD', items) if unicodedata.category(c) != 'Mn'))
        # Lemmatize words so we can compare words to measuring words
        items = [lemmatizer.lemmatize(word) for word in items]
        # Gets rid of measuring words/phrases, e.g. heaped teaspoon
        items = [word for word in items if word not in measures]
        # Get rid of common easy words
        items = [word for word in items if word not in words_to_remove]
        if items:
            ingred_list.append(' '.join(items)) 
    ingred_list = " ".join(ingred_list)
    return ingred_list


def createModel():
    showStatus("creating model")
    # parses the recipes into words
    recipe_df = load_data(RECIPES_PATH)
    recipe_df["desc"] = recipe_df['ingredients']
    # change the way the sentence is arranged in the data
    recipe_df['ingredients'] = recipe_df['ingredients'].map(str) + ',' + recipe_df['recipe_name'].map(str)
    recipe_df['ingredients'] = recipe_df['ingredients'].str.split()

    recipe_df['ingredients_parsed'] = recipe_df['ingredients'].apply(lambda x: ingredient_parser(x))

    df = recipe_df[['recipe_name', 'desc', 'ingredients_parsed', 'ingredients', 'recipe_urls']]
    df = recipe_df.dropna()

    # remove - Allrecipes.com from end of every recipe title 
    m = df.recipe_name.str.endswith('Recipe - Allrecipes.com')
    df['recipe_name'].loc[m] = df.recipe_name.loc[m].str[:-23]        
    #df.to_csv(PARSED_PATH, index=False) #save the parsed file
    save_data(df,PARSED_PATH)


def loadModel():
    showStatus("loading model")
    # load in parsed recipe dataset     
    df_recipes = load_data(PARSED_PATH)
    df_recipes['ingredients_parsed'] = df_recipes.ingredients_parsed.values.astype('U')

    # TF-IDF feature extractor 
    tfidf = TfidfVectorizer()
    tfidf.fit(df_recipes['ingredients_parsed'])
    tfidf_recipe = tfidf.transform(df_recipes['ingredients_parsed'])

    # save the tfidf model and encodings 
    with open(TFIDF_MODEL_PATH, "wb") as f:
        pickle.dump(tfidf, f)

    with open(TFIDF_ENCODING_PATH, "wb") as f:
        pickle.dump(tfidf_recipe, f)



def search_term_not_found(term):
    df = load_data(PARSED_PATH)
    return df[df['ingredients_parsed'].str.contains(term)]


def complementaryColor(color_choice):
    if color_choice=="red":
        return "green"
    elif color_choice=="blue":
        return "orange"
    elif color_choice=="orange":
        return "blue"
    elif color_choice=="yellow":
        return "purple"
    elif color_choice=="purple":
        return "red, blue"
    elif color_choice=="green":
        return "yellow, blue"
    elif color_choice=="orange":
        return "yellow, red"
    else:
        return "black"


#def letRecommend(search_term, number_of_rec):
def letRecommend(color_choice, fabric_choice):
    search_term = color_choice + " " + fabric_choice
    st.write("searching:" + search_term)
    results = RecSys(search_term, 1)
    write_results(results)
    
    color_choice = complementaryColor(color_choice)
    search_term = color_choice + " " + fabric_choice
    st.write("searching:" + search_term)
    results = RecSys(search_term, 2)
    write_results(results)


def write_results(results):
    for ind in results.index:
        col1, mid, col2 = st.columns([1,1,20])
        with col1:
                st.image(results['url'][ind], width=60)
        with col2:
                st.write(results['recipe'][ind])
                st.write(results['desc'][ind])
                score = "confidence :" + str(round(float(results['score'][ind])*100,0)) + "%"
                st.write(score)



def main():

    st.title("fashion recommender v3")
    menu = ["Home","Recommend", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    #createModel()
    loadModel()

    if choice == "Home":
        st.subheader("Home")
        st.write("please select on left to recommend")
    
    elif choice == "Recommend":
        st.subheader("Recommend")
        #search_term = st.text_input("Search")

        menu_color = ["red","blue","violet","green","pink", "indigo", "yellow", "orange", "white"]
        #color_choice = st.sidebar.selectbox("Color", menu_color)
        color_choice_2 = st.sidebar.color_picker("pick a color")
        menu_fabric = ["linen", "silk", "cotton"]
        fabric_choice = st.sidebar.selectbox("fabric", menu_fabric)
        #num_of_rec = st.sidebar.number_input("Number",1,10,7)
        if st.sidebar.button("Recommend"):
            #letRecommend(search_term, num_of_rec)
            color_choice = find_name(color_choice_2)
            st.write("color choice: " + color_choice)
            letRecommend(color_choice, fabric_choice)

    else:
        st.subheader("About")
        st.text("Built with Streamlit & Pandas")
        st.text("proof of concept")


if __name__ == "__main__":
    main()

