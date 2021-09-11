# this is the latest update to the recommender
# we will save the model and reuse instead of creating it all the time
# v3 Sep 2021

# run this program - streamlit run /home/ec2-user/project-python/StylePari/fashion_recommender_v3.py
#  
# Core Pkg
import streamlit as st 
import streamlit.components.v1 as stc 

import recommend_algorithm as rec
import color_functions as cf




#def letRecommend(search_term, number_of_rec):
def letRecommend(color_choice, fabric_choice):
    search_term = color_choice + " " + fabric_choice
    st.write("searching:" + search_term)
    results = rec.RecSys(search_term, 1)
    write_results(results)
    
    color_choice = cf.complementaryColor(color_choice)
    search_term = color_choice + " " + fabric_choice
    st.write("searching:" + search_term)
    results = rec.RecSys(search_term, 2)
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

    #rec.createModel()
    rec.loadModel()

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
            color_choice = cf.find_name(color_choice_2)
            st.write("color choice: " + color_choice)
            letRecommend(color_choice, fabric_choice)

    else:
        st.subheader("About")
        st.text("Built with Streamlit & Pandas")
        st.text("proof of concept")
        st.write("questions: talk to me")


if __name__ == "__main__":
    main()

