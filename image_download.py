
#utility to download image from website

import urllib.request
import pandas as pd

df = pd.read_csv ('./data/image_list.csv')
df.head()

# let us experiment downloading a small set of images
small_dataset = df.head(5)
small_dataset

file_name_old = "./images/download_filename_"
for ind in small_dataset.index:
     imgURL = small_dataset['Image Src'][ind]
     file_name = file_name_old + str(ind) + ".jpg"
     print(file_name)
     urllib.request.urlretrieve(imgURL, file_name)