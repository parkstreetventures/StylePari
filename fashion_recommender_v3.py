# this is the latest update to the recommender
# we will save the model and reuse instead of creating it all the time
# v3 Sep 2021

# Core Pkg
import streamlit as st 
import streamlit.components.v1 as stc 


# Load EDA
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


# nltk.download('wordnet')

# that's all the libaries that we need ..........

# path to all files
RECIPES_PATH = "/home/ec2-user/project-python/StylePari/data/saree_data.csv"
PARSED_PATH = "/home/ec2-user/project-python/StylePari/data/saree_parsed_new.csv"
TFIDF_ENCODING_PATH = "/home/ec2-user/project-python/StylePari/model/saree_tfidf_encodings.pkl"
TFIDF_MODEL_PATH = "/home/ec2-user/project-python/StylePari/model/saree_tfidf.pkl"


# debugging help file
def showStatus(moduleName):
    debug_mode="Y"
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
        recommendation.at[count, 'url'] = df_recipes['recipe_urls'][i]
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
    words_to_remove = ['(',')','.','\'','fresh', 'oil', 'a', 'and',  'or',  'large', 'extra',  'free', 'small', 'from', 'higher', 'for', 'finely', 'freshly', 'to', 'organic', 'the', 'plain', 'plus' ]
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


def letRecommend(search_term, number_of_rec):
    if search_term is not None:
        try:
            # add a module to modify search ..
            results = RecSys(search_term, number_of_rec)
            print(results)
        except:
            results = "not found"
            st.warning(results)
            # maybe recommend similar results as options


def main():

    st.title("fashion recommender v3")
    menu = ["Home","Recommend", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    createModel()
    loadModel()

    if choice == "Home":
        st.subheader("Home")
        st.write("please select on left to recommend")
    
    elif choice == "Recommend":
        st.subheader("Recommend")
        search_term = st.text_input("Search")
        num_of_rec = st.sidebar.number_input("Number",4,10,7)
        if st.button("Recommend"):
            letRecommend(search_term, num_of_rec)

    else:
        st.subheader("About")
        st.text("Built with Streamlit & Pandas")
        st.text("proof of concept")


if __name__ == "__main__":
    main()

