import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error




data = pd.read_csv('C:/Users/pc/Documents/air quality/cars.csv')
print(data.head())

num_classes = 20  # Par exemple, diviser en 5 classes
data['income_binned'] = pd.cut(data['income'], bins=num_classes)  # Crée les classes automatiquement

# Variables nécessitant un groupby
groupby_features = ['miles', 'income_binned', 'debt']

# Variables sans groupby
no_groupby_features = ['gender', 'age']

# Visualisation avec groupby
for feature in groupby_features:
    grouped_data = data.groupby(feature)['sales'].mean()  # Moyenne des ventes par catégorie
    plt.figure(figsize=(8, 6))
    grouped_data.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Sales par {feature}')
    plt.ylabel('Sales (moyenne)')
    plt.xlabel(feature)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Visualisation sans groupby
for feature in no_groupby_features:
    plt.figure(figsize=(8, 6))
    data[feature].value_counts().plot(kind='bar', color='coral', edgecolor='black')
    plt.title(f'Distribution de {feature}')
    plt.ylabel('Nombre d\'occurrences')
    plt.xlabel(feature)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Variables indépendantes (features) et dépendantes (cible)

print(data.describe())

# Normalize 'miles'
miles_min = data['miles'].min()
miles_max = data['miles'].max()
data['miles'] = (data['miles'] - miles_min) / (miles_max - miles_min)

# Normalize 'income'
income_min = data['income'].min()
income_max = data['income'].max()
data['income'] = (data['income'] - income_min) / (income_max - income_min)

# Normalize 'debt'
debt_min = data['debt'].min()
debt_max = data['debt'].max()
data['debt'] = (data['debt'] - debt_min) / (debt_max - debt_min)

# Normalize 'sales'
sales_min = data['sales'].min()
sales_max = data['sales'].max()
data['sales'] = (data['sales'] - sales_min) / (sales_max - sales_min)

# Split data into features and target
x = data[['age', 'gender', 'miles', 'debt', 'income']]  # Features
y = data['sales']  # Target variable

# Print final data statistics
print(data.head())

k = 10 
fold_size = len(x) // k

for i in range(k):
    test_start = i * fold_size
    test_end = min((i + 1) * fold_size, len(x))

    # Diviser les données en ensemble d'entraînement et de test
    X_test = x.iloc[test_start:test_end]
    y_test = y.iloc[test_start:test_end]

    X_train = pd.concat([x.iloc[:test_start], x.iloc[test_end:]])
    y_train = pd.concat([y.iloc[:test_start], y.iloc[test_end:]])

    # Afficher pour vérification
    print(f"Fold {i + 1}/{k}")
    print(f"Test indices: {test_start} to {test_end - 1}")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print("-" * 50)


feature = 'age'
X_train_feature = X_train[feature]  # Select 'Year' column

    # Calculer les moyennes
X_mean = np.mean(X_train_feature)
y_mean = np.mean(y_train)

    # Calculer la pente (a)
numerator = np.sum((X_train_feature - X_mean) * (y_train - y_mean))
denominator = np.sum((X_train_feature - X_mean) ** 2)
a = numerator / denominator
print(f"Pente (a): {a}")

    # Calculer l'ordonnée à l'origine (b)
b = y_mean - a * X_mean
print(f"Ordonnée à l'origine (b): {b}")

    # Prédictions sur l'ensemble de test
X_test_feature = X_test[feature]
y_pred = a * X_test_feature + b

    # Calcul des résidus
residuals = y_test - y_pred

    # Calcul des résidus au carré
squared_residuals = residuals ** 2

    # Calcul du MSE pour cette variable
mse = np.mean(squared_residuals)
print(f"MSE for {feature}: {mse:.4f}")

    # Affichage graphique de la régression
plt.figure(figsize=(8, 5))
plt.scatter(x['age'], y, color='blue', label='Test Data')  # Données réelles
plt.plot(X_test_feature, y_pred, color='red', label='Regression Line')  # Ligne de régression
plt.xlabel(feature)
plt.ylabel("sales")
plt.title(f"Linear Regression for {feature} Prediction")
plt.legend()
plt.show() 
    

feature = 'gender'
X_train_feature = X_train[feature]  # Select 'Year' column

    # Calculer les moyennes
X_mean = np.mean(X_train_feature)
y_mean = np.mean(y_train)

    # Calculer la pente (a)
numerator = np.sum((X_train_feature - X_mean) * (y_train - y_mean))
denominator = np.sum((X_train_feature - X_mean) ** 2)
a = numerator / denominator
print(f"Pente (a): {a}")

    # Calculer l'ordonnée à l'origine (b)
b = y_mean - a * X_mean
print(f"Ordonnée à l'origine (b): {b}")

    # Prédictions sur l'ensemble de test
X_test_feature = X_test[feature]
y_pred = a * X_test_feature + b

    # Calcul des résidus
residuals = y_test - y_pred

    # Calcul des résidus au carré
squared_residuals = residuals ** 2

    # Calcul du MSE pour cette variable
mse = np.mean(squared_residuals)
print(f"MSE for {feature}: {mse:.4f}")

    # Affichage graphique de la régression
plt.figure(figsize=(8, 5))
plt.scatter(x['gender'], y, color='blue', label='Test Data')  # Données réelles
plt.plot(X_test_feature, y_pred, color='red', label='Regression Line')  # Ligne de régression
plt.xlabel(feature)
plt.ylabel("sales")
plt.title(f"Linear Regression for {feature} Prediction")
plt.legend()
plt.show() 


feature = 'miles'
X_train_feature = X_train[feature]  # Select 'Year' column

    # Calculer les moyennes
X_mean = np.mean(X_train_feature)
y_mean = np.mean(y_train)

    # Calculer la pente (a)
numerator = np.sum((X_train_feature - X_mean) * (y_train - y_mean))
denominator = np.sum((X_train_feature - X_mean) ** 2)
a = numerator / denominator
print(f"Pente (a): {a}")

    # Calculer l'ordonnée à l'origine (b)
b = y_mean - a * X_mean
print(f"Ordonnée à l'origine (b): {b}")

    # Prédictions sur l'ensemble de test
X_test_feature = X_test[feature]
y_pred = a * X_test_feature + b

    # Calcul des résidus
residuals = y_test - y_pred

    # Calcul des résidus au carré
squared_residuals = residuals ** 2

    # Calcul du MSE pour cette variable
mse = np.mean(squared_residuals)
print(f"MSE for {feature}: {mse:.4f}")

    # Affichage graphique de la régression
plt.figure(figsize=(8, 5))
plt.scatter(x['miles'], y, color='blue', label='Test Data')  # Données réelles
plt.plot(X_test_feature, y_pred, color='red', label='Regression Line')  # Ligne de régression
plt.xlabel(feature)
plt.ylabel("sales")
plt.title(f"Linear Regression for {feature} Prediction")
plt.legend()
plt.show() 
    

feature = 'debt'
X_train_feature = X_train[feature]  # Select 'Year' column

    # Calculer les moyennes
X_mean = np.mean(X_train_feature)
y_mean = np.mean(y_train)

    # Calculer la pente (a)
numerator = np.sum((X_train_feature - X_mean) * (y_train - y_mean))
denominator = np.sum((X_train_feature - X_mean) ** 2)
a = numerator / denominator
print(f"Pente (a): {a}")

    # Calculer l'ordonnée à l'origine (b)
b = y_mean - a * X_mean
print(f"Ordonnée à l'origine (b): {b}")

    # Prédictions sur l'ensemble de test
X_test_feature = X_test[feature]
y_pred = a * X_test_feature + b

    # Calcul des résidus
residuals = y_test - y_pred

    # Calcul des résidus au carré
squared_residuals = residuals ** 2

    # Calcul du MSE pour cette variable
mse = np.mean(squared_residuals)
print(f"MSE for {feature}: {mse:.4f}")

    # Affichage graphique de la régression
plt.figure(figsize=(8, 5))
plt.scatter(x['debt'], y, color='blue', label='Test Data')  # Données réelles
plt.plot(X_test_feature, y_pred, color='red', label='Regression Line')  # Ligne de régression
plt.xlabel(feature)
plt.ylabel("sales")
plt.title(f"Linear Regression for {feature} Prediction")
plt.legend()
plt.show() 

feature = 'income'
X_train_feature = X_train[feature]  # Select 'Year' column

    # Calculer les moyennes
X_mean = np.mean(X_train_feature)
y_mean = np.mean(y_train)

    # Calculer la pente (a)
numerator = np.sum((X_train_feature - X_mean) * (y_train - y_mean))
denominator = np.sum((X_train_feature - X_mean) ** 2)
a = numerator / denominator
print(f"Pente (a): {a}")

    # Calculer l'ordonnée à l'origine (b)
b = y_mean - a * X_mean
print(f"Ordonnée à l'origine (b): {b}")

    # Prédictions sur l'ensemble de test
X_test_feature = X_test[feature]
y_pred = a * X_test_feature + b

    # Calcul des résidus
residuals = y_test - y_pred

    # Calcul des résidus au carré
squared_residuals = residuals ** 2

    # Calcul du MSE pour cette variable
mse = np.mean(squared_residuals)
print(f"MSE for {feature}: {mse:.4f}")

    # Affichage graphique de la régression
plt.figure(figsize=(8, 5))
plt.scatter(x['income'], y, color='blue', label='Test Data')  # Données réelles
plt.plot(X_test_feature, y_pred, color='red', label='Regression Line')  # Ligne de régression
plt.xlabel(feature)
plt.ylabel("sales")
plt.title(f"Linear Regression for {feature} Prediction")
plt.legend()
plt.show() 


x_with_intercept = np.hstack([np.ones((x.shape[0], 1)), x])
x_with_intercept_df = pd.DataFrame(x_with_intercept)

print(x_with_intercept_df.head())

x_transposed = x_with_intercept_df.T
print(x_transposed.head())

x_inverse = np.linalg.inv(x_transposed @ x_transposed.T)
print("Inverse de la matrice calculée avec succès:")
print(x_inverse)

vecteur_parametres = x_inverse @ x_transposed @ y
print(vecteur_parametres)




def polynomial_model(age, gender, miles, debt, income):
    beta = np.array([-0.103804, 0.002907, -0.001109, 0.347985, 0.570271,0.327234])
    
    return (beta[0] + beta[1] * age + beta[2] * gender + beta[3] * miles + beta[4] * debt + beta[5]*income )




y_predd = polynomial_model(2018, 6 ,12,17,2000 )

print(y_predd)

#modele
 
y_predii =  polynomial_model(X_test['age'], X_test['gender'] ,X_test['miles'], X_test['debt'],X_test['income'] )

#mse

residuals = y_test - y_predii

print(residuals)

# Step 5: Calculate the squared residuals
squared_residuals = residuals ** 2

# Step 6: Calculate the Mean Squared Error (MSE) for this feature
msee = np.mean(squared_residuals)

# Print the MSE for this feature
print(f"mse : {msee}")


n = len(X_test)

for column in X_test.columns:
    # Vérifier si la colonne a une variance nulle
    if np.var(X_test[column]) == 0:
        print(f"La colonne {column} contient des valeurs constantes. Modèle non applicable.")
        continue

    # Construire la matrice M
    M = np.array([
        [np.sum(X_test[column]**4), np.sum(X_test[column]**3), np.sum(X_test[column]**2)],
        [np.sum(X_test[column]**3), np.sum(X_test[column]**2), np.sum(X_test[column])],
        [np.sum(X_test[column]**2), np.sum(X_test[column]), n]
    ])

    # Vérifier si M est inversible
    if np.linalg.det(M) == 0:
        print(f"La matrice M pour la variable {column} est singulière et ne peut pas être inversée.")
        continue

    # Construire le vecteur Y
    Y = np.array([
        np.sum(X_test[column]**2 * y_test),
        np.sum(X_test[column] * y_test),
        np.sum(y_test)
    ])

    # Calculer les coefficients avec régularisation si nécessaire
   # lambda_ = 1e-5  # Terme de régularisation
    #M = M + lambda_ * np.eye(M.shape[0])
    coefficients = np.linalg.inv(M) @ Y

    print(f"Variable: {column}")
    print(f"Model coefficients: {coefficients[0]}, {coefficients[1]}, {coefficients[2]}")

    # Calculer les prédictions
    y_pred = coefficients[0] * (X_test[column] ** 2) + coefficients[1] * X_test[column] + coefficients[2]

    # Calculer le MSE
    mse = np.mean((y_test - y_pred) ** 2)
    print(f"Mean Squared Error (MSE): {mse}\n")

    # Dessiner le graphe
    plt.figure()
    plt.scatter(X_test[column], y_test, color='blue', label='Données réelles')  # Points réels
    variable_range = np.linspace(min(X_test[column]), max(X_test[column]), 100)
    y_model = coefficients[0] * (variable_range  ** 2) + coefficients[1] * variable_range  + coefficients[2]
    plt.plot(variable_range, y_model, color='red', label='Modèle quadratique')  # Courbe du modèle

    # Ajouter des labels et une légende
    plt.xlabel(column)
    plt.ylabel('Target')
    plt.title(f'Modèle de régression quadratique pour {column}')
    plt.legend()

    # Afficher le graphe
    plt.show()
