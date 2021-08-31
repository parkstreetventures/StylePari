from matplotlib import colors

color_name = 'red'
color_rgb = colors.hex2color(colors.cnames[color_name])

print(color_rgb)

#from webcolors import rgb_to_name

#named_color = rgb_to_name((color_rgb), spec='css3')

#print(named_color)

from scipy.spatial import KDTree

import webcolors
#from webcolors import css3_hex_to_names

#from webcolors import hex_to_rgb,

def convert_rgb_to_names(rgb_tuple):
    
    # a dictionary of all the hex and their respective names in css3
    css3_db = webcolors.CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(webcolors.hex_to_rgb(color_hex))
    
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return f'closest match: {names[index]}'


print(convert_rgb_to_names((color_rgb)))

import matplotlib

print(matplotlib.colors.to_rgb(color_name))

print(matplotlib.colors.get_named_colors_mapping())