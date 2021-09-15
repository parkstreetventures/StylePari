# this is the latest update to the recommender
# we will save the model and reuse instead of creating it all the time
# v3 Sep 2021

# run this program - streamlit run /Users/sangames/Documents/python-dev/StylePari/fashion_recommender_v3.py
#  
# Core Pkg
import streamlit as st 
import streamlit.components.v1 as stc 

import fashion_recommender_algorithm as rec

from pathlib import Path


def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()


def goRecommend(color_choice, fabric_choice, pattern_choice, sleeve_choice, strap_choice):
    results = rec.recommendationEngine(color_choice, fabric_choice, pattern_choice, sleeve_choice, strap_choice)
    #st.write(results)
    results.set_index('title', inplace=True)
    #st.dataframe(results)
    write_results(results)


def write_results(results):
    for ind in results.index:
        #score = "confidence :" + str(round(float(results['score'][1])*100,0)) + "%"
        #score="0"
        col1, mid, col2 = st.columns([1,1,20])

        with col1:
                st.image(str(results['url'][ind]), width=60)
                #st.write("hello")
        with col2:
                score = results['score'][ind]
                st.write(ind)
                #st.write(results['title'][ind])
                st.write(results['desc'][ind])
                st.write(results['color'][ind])
                st.write("confidence :" + str(round(float(score)*100,0)) + "%")



def home():
    st.subheader("Home")
    intro_markdown = read_markdown_file("introduction.md")
    st.markdown(intro_markdown, unsafe_allow_html=True)


def about():
    st.subheader("About")
    st.text("Built with Streamlit & Pandas")
    st.text("proof of concept")
    st.write("questions: talk to me")


def recommender():
    st.subheader("Recommend")
    #search_term = st.text_input("Search")
  
    color_choice = st.sidebar.color_picker('pick a color')
    fabric_choice = st.sidebar.selectbox("fabric", ["linen", "silk", "cotton"])
    pattern_choice = st.sidebar.selectbox("pattern", ["plain","stripes","random"])
    sleeve_choice = st.sidebar.radio('sleeve', ["no sleeve", "full sleeve","normal"])
    strap_choice = st.sidebar.selectbox("strap", ["strapless", "bare back"])

    #num_of_rec = st.sidebar.number_input("Number",1,10,7)
    if st.sidebar.button("Recommend"):
        goRecommend(color_choice, fabric_choice, pattern_choice, sleeve_choice, strap_choice)


def main():

    st.title("fashion recommender v3")
    menu = ["Home","Recommend", "About"]
    choice = st.sidebar.selectbox("Menu", menu)


    if choice == "Home":
        home()
    
    elif choice == "Recommend":
        recommender()

    else:
        about()


if __name__ == "__main__":
    main()

