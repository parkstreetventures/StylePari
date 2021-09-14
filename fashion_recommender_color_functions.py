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



import random 

def complementaryColor(color_choice):
    random_choice = ['red','blue','black','white','green','olive']
    return random.choice(random_choice)

# end color functionality