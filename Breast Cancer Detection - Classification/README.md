# Breast Cancer Detection - Classification - Pycaret

**Complete Video Tutorial:** https://youtu.be/janK8osSKPs

# Dataset Information

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

## Attribute Information:

1) ID number
2) Diagnosis (M = malignant, B = benign)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)

The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image,
resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.

Missing attribute values: none

Class distribution: 357 benign, 212 malignant


**Download link:** https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

# Libraries

- pandas
- matplotlib
- seaborn
- pycaret

# Algorithms

- Logistic Regression
- Decision Tree
- Random Forest
- Extra Tress
- XGBoost
- LightGBM
- CatBoost
  
**Best Model AUC:** 99.47