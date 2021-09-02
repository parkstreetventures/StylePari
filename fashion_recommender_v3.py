# this is the latest update to the recommender
# we will save the model and reuse instead of creating it all the time
# v3 Sep 2021

# Core Pkg
import streamlit as st 
import streamlit.components.v1 as stc 


# Load EDA
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
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

# begin original code to modify

# path to all files
RECIPES_PATH = "/home/ec2-user/project-python/StylePari/data/saree_data.csv"
PARSED_PATH = "/home/ec2-user/project-python/StylePari/data/saree_parsed_new.csv"
TFIDF_ENCODING_PATH = "/home/ec2-user/project-python/StylePari/model/saree_tfidf_encodings.pkl"
TFIDF_MODEL_PATH = "/home/ec2-user/project-python/StylePari/model/saree_tfidf.pkl"

def showStatus(moduleName):
    debug_mode="Y"
    if debug_mode=="Y":
        print(moduleName)

def path_to_image_html(path):
    return '<img src="'+ path + '" width="80" >'

# Weigths and measures are words that will not add value to the model. I got these standard words from 
# https://en.wikibooks.org/wiki/Cookbook:Units_of_measurement

# # We lemmatize the words to reduce them to their smallest form (lemmas). 
# lemmatizer = WordNetLemmatizer()
# measures = [lemmatizer.lemmatize(m) for m in measures]
# words_to_remove = [lemmatizer.lemmatize(m) for m in words_to_remove]

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
    measures = ['teaspoon', 't', 'tsp.', 'tablespoon', 'T', 'tbl.', 'tb', 'tbsp.', 'fluid ounce', 'fl oz', 'gill', 'cup', 'c', 'pint', 'p', 'pt', 'fl pt', 'quart', 'q', 'qt', 'fl qt', 'gallon', 'g', 'gal', 'ml', 'milliliter', 'millilitre', 'cc', 'mL', 'l', 'liter', 'litre', 'L', 'dl', 'deciliter', 'decilitre', 'dL', 'bulb', 'level', 'heaped', 'rounded', 'whole', 'pinch', 'medium', 'slice', 'pound', 'lb', '#', 'ounce', 'oz', 'mg', 'milligram', 'milligramme', 'g', 'gram', 'gramme', 'kg', 'kilogram', 'kilogramme', 'x', 'of', 'mm', 'millimetre', 'millimeter', 'cm', 'centimeter', 'centimetre', 'm', 'meter', 'metre', 'inch', 'in', 'milli', 'centi', 'deci', 'hecto', 'kilo']
    words_to_remove = ['(',')','.','\'','fresh', 'oil', 'a', 'red', 'bunch', 'and', 'clove', 'or', 'leaf', 'chilli', 'large', 'extra', 'sprig', 'ground', 'handful', 'free', 'small', 'pepper', 'virgin', 'range', 'from', 'dried', 'sustainable', 'black', 'peeled', 'higher', 'welfare', 'seed', 'for', 'finely', 'freshly', 'sea', 'quality', 'white', 'ripe', 'few', 'piece', 'source', 'to', 'organic', 'flat', 'smoked', 'ginger', 'sliced', 'green', 'picked', 'the', 'stick', 'plain', 'plus', 'mixed', 'mint', 'bay', 'basil', 'your', 'cumin', 'optional', 'fennel', 'serve', 'mustard', 'unsalted', 'baby', 'paprika', 'fat', 'ask', 'natural', 'skin', 'roughly', 'into', 'such', 'cut', 'good', 'brown', 'grated', 'trimmed', 'oregano', 'powder', 'yellow', 'dusting', 'knob', 'frozen', 'on', 'deseeded', 'low', 'runny', 'balsamic', 'cooked', 'streaky', 'nutmeg', 'sage', 'rasher', 'zest', 'pin', 'groundnut', 'breadcrumb', 'turmeric', 'halved', 'grating', 'stalk', 'light', 'tinned', 'dry', 'soft', 'rocket', 'bone', 'colour', 'washed', 'skinless', 'leftover', 'splash', 'removed', 'dijon', 'thick', 'big', 'hot', 'drained', 'sized', 'chestnut', 'watercress', 'fishmonger', 'english', 'dill', 'caper', 'raw', 'worcestershire', 'flake', 'cider', 'cayenne', 'tbsp', 'leg', 'pine', 'wild', 'if', 'fine', 'herb', 'almond', 'shoulder', 'cube', 'dressing', 'with', 'chunk', 'spice', 'thumb', 'garam', 'new', 'little', 'punnet', 'peppercorn', 'shelled', 'saffron', 'other''chopped', 'salt', 'olive', 'taste', 'can', 'sauce', 'water', 'diced', 'package', 'italian', 'shredded', 'divided', 'parsley', 'vinegar', 'all', 'purpose', 'crushed', 'juice', 'more', 'coriander', 'bell', 'needed', 'thinly', 'boneless', 'half', 'thyme', 'cubed', 'cinnamon', 'cilantro', 'jar', 'seasoning', 'rosemary', 'extract', 'sweet', 'baking', 'beaten', 'heavy', 'seeded', 'tin', 'vanilla', 'uncooked', 'crumb', 'style', 'thin', 'nut', 'coarsely', 'spring', 'chili', 'cornstarch', 'strip', 'cardamom', 'rinsed', 'honey', 'cherry', 'root', 'quartered', 'head', 'softened', 'container', 'crumbled', 'frying', 'lean', 'cooking', 'roasted', 'warm', 'whipping', 'thawed', 'corn', 'pitted', 'sun', 'kosher', 'bite', 'toasted', 'lasagna', 'split', 'melted', 'degree', 'lengthwise', 'romano', 'packed', 'pod', 'anchovy', 'rom', 'prepared', 'juiced', 'fluid', 'floret', 'room', 'active', 'seasoned', 'mix', 'deveined', 'lightly', 'anise', 'thai', 'size', 'unsweetened', 'torn', 'wedge', 'sour', 'basmati', 'marinara', 'dark', 'temperature', 'garnish', 'bouillon', 'loaf', 'shell', 'reggiano', 'canola', 'parmigiano', 'round', 'canned', 'ghee', 'crust', 'long', 'broken', 'ketchup', 'bulk', 'cleaned', 'condensed', 'sherry', 'provolone', 'cold', 'soda', 'cottage', 'spray', 'tamarind', 'pecorino', 'shortening', 'part', 'bottle', 'sodium', 'cocoa', 'grain', 'french', 'roast', 'stem', 'link', 'firm', 'asafoetida', 'mild', 'dash', 'boiling']
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

# recommendation system

# Top-N recomendations order by score
def get_recommendations(N, scores):
    # load in recipe dataset 
    df_recipes = pd.read_csv(PARSED_PATH)
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

    # parse the ingredients using my ingredient_parser 
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

def createModel():
    showStatus("creating model")
    # parses the recipes into words
    recipe_df = pd.read_csv(RECIPES_PATH)
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
    df.to_csv(PARSED_PATH, index=False) #save the parsed file


def loadModel():
    showStatus("loding model")
    # load in parsed recipe dataset     
    df_recipes = pd.read_csv(PARSED_PATH)
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

# end original code to modify


# Load Our Dataset
def load_data(data):
	df = pd.read_csv(data)
	return df 


# Fxn
# Vectorize + Cosine Similarity Matrix

def vectorize_text_to_cosine_mat(data):
	count_vect = CountVectorizer()
	cv_mat = count_vect.fit_transform(data)
	# Get the cosine
	cosine_sim_mat = cosine_similarity(cv_mat)
	return cosine_sim_mat



# Recommendation Sys
@st.cache
def get_recommendation(title,cosine_sim_mat,df,num_of_rec=10):
	# indices of the course
	course_indices = pd.Series(df.index,index=df['ingredients']).drop_duplicates()
	# Index of course
	idx = course_indices[title]

	# Look into the cosine matr for that index
	sim_scores =list(enumerate(cosine_sim_mat[idx]))
	sim_scores = sorted(sim_scores,key=lambda x: x[1],reverse=True)
	selected_course_indices = [i[0] for i in sim_scores[1:]]
	selected_course_scores = [i[0] for i in sim_scores[1:]]

	# Get the dataframe & title
	result_df = df.iloc[selected_course_indices]
	result_df['similarity_score'] = selected_course_scores
	final_recommended_courses = result_df[['recipe_name','similarity_score','recipe_urls','ingredients']]
	return final_recommended_courses.head(num_of_rec)


RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #a8f0c6;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">📈Score::</span>{}</p>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}",target="_blank">Link</a></p>
<p style="color:blue;"><span style="color:black;">URL:</span>{}</p>
<p style="color:blue;"><span style="color:black;">🧑‍🎓👨🏽‍🎓 Ingredients:</span>{}</p>
</div>
"""

# Search For Course 
@st.cache
def search_term_if_not_found(term,df):
	result_df = df[df['ingredients'].str.contains(term)]
	return result_df


def main():

    # -- need to modify this
    
    #showStatus("main")
    #createModel()
    #loadModel()
    # test the system
    #test_ingredients = "cotton mulmul"
    #recs = RecSys(test_ingredients)
    #print(recs.score)
    #print(recs)

    # -- end need to modify this

	st.title("Course Recommendation App")

	menu = ["Home","Recommend","About"]
	choice = st.sidebar.selectbox("Menu",menu)

	df = load_data("/home/ec2-user/project-python/StylePari/data/saree_data.csv")

	if choice == "Home":
		st.subheader("Home")
		st.dataframe(df.head(10))


	elif choice == "Recommend":
		st.subheader("Recommend Courses")
		cosine_sim_mat = vectorize_text_to_cosine_mat(df['ingredients'])
		search_term = st.text_input("Search")
		num_of_rec = st.sidebar.number_input("Number",4,30,7)
		if st.button("Recommend"):
			if search_term is not None:
				try:
					results = get_recommendation(search_term,cosine_sim_mat,df,num_of_rec)
					with st.beta_expander("Results as JSON"):
						results_json = results.to_dict('index')
						st.write(results_json)

					for row in results.iterrows():
						rec_title = row[1][0]
						rec_score = row[1][1]
						rec_url = row[1][2]
						#rec_price = row[1][2]
						rec_num_sub = row[1][3]

						# st.write("Title",rec_title,)
						stc.html(RESULT_TEMP.format(rec_title,rec_score,rec_url,rec_num_sub),height=350)
				except:
					results= "Not Found"
					st.warning(results)
					st.info("Suggested Options include")
					result_df = search_term_if_not_found(search_term,df)
					st.dataframe(result_df)



				# How To Maximize Your Profits Options Trading




	else:
		st.subheader("About")
		st.text("Built with Streamlit & Pandas")


if __name__ == '__main__':
	main()