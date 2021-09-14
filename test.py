
import fashion_recommender_algorithm as rec

color_choice="#730b28"
fabric_choice = "cotton"

y = rec.recommendationEngine(color_choice, fabric_choice)

print(type(y))
print(y)

