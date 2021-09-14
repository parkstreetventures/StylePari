# this is the latest update to the recommender
# we will save the model and reuse instead of creating it all the time
# v3 Sep 2021

# run this program - streamlit run /home/ec2-user/project-python/StylePari/fashion_recommender_v3.py
#  
# Core Pkg
import streamlit as st 
import streamlit.components.v1 as stc 

import fashion_recommender_algorithm as rec

from pathlib import Path


def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()


def goRecommend(color_choice, fabric_choice):
    results = rec.recommendationEngine(color_choice, fabric_choice)
    st.write(results)
    #write_results(results)


def write_results(results):
    for ind in results.index:
        #score = "confidence :" + str(round(float(results['score'][ind])*100,0)) + "%"
        score="0"
        col1, mid, col2 = st.columns([1,1,20])

        with col1:
                #st.image(results['url'][ind], width=60)
                st.write("hello")
        with col2:
                st.write(ind)
                st.write(results['title'])
                st.write(results['desc'][ind])
                st.write(results['color'][ind])
                st.write(score)


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

    #menu_color = ["red","blue","violet","green","pink", "indigo", "yellow", "orange", "white"]
    #color_choice = st.sidebar.selectbox("Color", menu_color)
    color_choice = st.sidebar.color_picker("pick a color")
    menu_fabric = ["linen", "silk", "cotton"]
    fabric_choice = st.sidebar.selectbox("fabric", menu_fabric)
    #num_of_rec = st.sidebar.number_input("Number",1,10,7)
    if st.sidebar.button("Recommend"):
        goRecommend(color_choice, fabric_choice)


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

