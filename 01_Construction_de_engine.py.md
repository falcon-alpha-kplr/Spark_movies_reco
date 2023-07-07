**Étape 1: Importation des bibliothèques nécessaires**

```python
from pyspark.sql.types import *
from pyspark.sql.functions import explode, col
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SQLContext
```

- `pyspark.sql.types` : Importe les types de données nécessaires pour définir la structure du schéma des ensembles de données.
- `pyspark.sql.functions` : Importe les fonctions nécessaires pour effectuer des opérations sur les colonnes des ensembles de données.
- `pyspark.ml.recommendation.ALS` : Importe l'algorithme ALS (Alternating Least Squares) utilisé pour le filtrage collaboratif.
- `pyspark.ml.evaluation.RegressionEvaluator` : Importe l'évaluateur de régression utilisé pour évaluer les performances du modèle.
- `pyspark.sql.SQLContext` : Importe le contexte SQL utilisé pour créer une session Spark.

**Étape 2: Définition de la classe RecommendationEngine**

**Méthode create_user(self, user_id) :**

- Cette méthode permet de créer un nouvel utilisateur.
- Elle prend en paramètre un `user_id` facultatif pour spécifier l'identifiant de l'utilisateur. Si `user_id` est `None`, un nouvel identifiant est généré automatiquement.
- Si `user_id` est supérieur à `max_user_identifier`, `max_user_identifier` est mis à jour avec la valeur de `user_id`.
- La méthode retourne l'identifiant de l'utilisateur créé ou mis à jour.

**Méthode is_user_known(self, user_id) :**

- Cette méthode permet de vérifier si un utilisateur est connu.
- Elle prend en paramètre un `user_id` et retourne `True` si l'utilisateur est connu (c'est-à-dire si `user_id` est différent de `None` et inférieur ou égal à `max_user_identifier`), sinon elle retourne `False`.

**Méthode get_movie(self, movie_id) :**

- Cette méthode permet d'obtenir un film.
- Elle prend en paramètre un `movie_id` facultatif pour spécifier l'identifiant du film. Si `movie_id` est `None`, la méthode retourne un échantillon aléatoire d'un film à partir du dataframe `best_movies_df`. Sinon, elle filtre le dataframe `movies_df` pour obtenir le film correspondant à `movie_id`.
- La méthode retourne un dataframe contenant les informations du film (colonne "movieId" et "title").

**Méthode get_ratings_for_user(self, user_id) :**

- Cette méthode permet d'obtenir les évaluations d'un utilisateur.
- Elle prend en paramètre un `user_id` et filtre le dataframe `ratings_df` pour obtenir les évaluations correspondantes à l'utilisateur.
- La méthode retourne un dataframe contenant les évaluations de l'utilisateur (colonnes "movieId", "userId" et "rating").

**Méthode add_ratings(self, user_id, ratings) :**

- Cette méthode permet d'ajouter de nouvelles évaluations au modèle et de re-entraîner le modèle.
- Elle prend en paramètres un `user_id` et une liste de `ratings` contenant les nouvelles évaluations.
- La méthode crée un nouveau dataframe `new_ratings_df` à partir de la liste de `ratings` et l'ajoute au dataframe existant `ratings_df` en utilisant l'opération `union()`.
- Ensuite, les données sont divisées en ensembles d'entraînement (`training`) et de test (`test`) en utilisant la méthode `randomSplit()`.
- Enfin, la méthode privée `__train_model()` est appelée pour re-entraîner le modèle.

**Méthode predict_rating(self, user_id, movie_id) :**

- Cette méthode permet de prédire une évaluation pour un utilisateur et un film donnés.
- Elle prend en paramètres un `user_id` et un `movie_id`.
- La méthode crée un dataframe `rating_df` à partir des données (`user_id`, `movie_id`) et le transforme en utilisant le modèle pour obtenir les prédictions.
- Si le dataframe de prédiction est vide, la méthode retourne `-1`, sinon elle retourne la valeur de prédiction.

**Méthode recommend_for_user(self, user_id, nb_movies) :**

- Cette méthode permet d'obtenir les meilleures recommandations pour un utilisateur donné.
- Elle prend en paramètres un `user_id` et un nombre de films `nb_movies` à recommander.
- La méthode crée un dataframe `user_df` contenant l'identifiant de l'utilisateur et utilise la méthode `recommendForUserSubset()` du modèle pour obtenir les recommandations pour cet utilisateur.
- Les recommandations sont ensuite jointes avec le dataframe `movies_df` pour obtenir les détails des films recommandés.
- Le dataframe résultant est retourné avec les colonnes "title" et d'autres colonnes du dataframe `movies_df`.

**Méthode __train_model(self) :**

- Cette méthode privée permet d'entraîner le modèle avec l'algorithme ALS (Alternating Least Squares).
- Elle utilise les paramètres `maxIter` et `regParam` définis dans l'initialisation de la classe pour créer une instance de l'algorithme ALS.
- Ensuite, le modèle est entraîné en utilisant le dataframe `training`.
- La méthode privée `__evaluate()` est appelée pour évaluer les performances du modèle.

**Méthode __evaluate(self) :**

- Cette méthode privée permet d'évaluer le modèle en calculant l'erreur quadratique moyenne (RMSE - Root-mean-square error).
- Elle utilise le modèle pour prédire les évaluations sur le dataframe `test`.
- Ensuite, elle utilise l'évaluateur de régression pour calculer le RMSE en comparant les prédictions avec les vraies évaluations.
- La valeur de RMSE est stockée dans la variable `rmse` de la classe et affichée à l'écran.

**Méthode **init**(self, sc, movies_set_path, ratings_set_path) :**

- Cette méthode d'initialisation est appelée lors de la création d'une instance de la classe RecommendationEngine.
- Elle prend en paramètres le contexte Spark (`sc`), le chemin vers l'ensemble de données de films (`movies_set_path`) et le chemin vers l'ensemble de données d'évaluations (`ratings_set_path`).
- La méthode initialise le contexte SQL à partir du contexte Spark, charge les données des ensembles de films et d'évaluations à partir des fichiers CSV spécifiés, définit le schéma des données, effectue diverses opérations de traitement des données et entraîne le modèle en utilisant la méthode privée `__train_model()`.

Voici un code initiale :

```python
class RecommendationEngine:
    def create_user(self, user_id):
        # Méthode pour créer un nouvel utilisateur
        ...

    def is_user_known(self, user_id):
        # Méthode pour vérifier si un utilisateur est connu
        ...

    def get_movie(self, movie_id):
        # Méthode pour obtenir un film
        ...

    def get_ratings_for_user(self, user_id):
        # Méthode pour obtenir les évaluations d'un utilisateur
        ...

    def add_ratings(self, user_id, ratings):
        # Méthode pour ajouter de nouvelles évaluations et re-entraîner le modèle
        ...

    def predict_rating(self, user_id, movie_id):
        # Méthode pour prédire une évaluation pour un utilisateur et un film donnés
        ...

    def recommend_for_user(self, user_id, nb_movies):
        # Méthode pour obtenir les meilleures recommandations pour un utilisateur donné
        ...

    def __train_model(self):
        # Méthode privée pour entraîner le modèle avec ALS
        ...

    def __evaluate(self):
        # Méthode privée pour évaluer le modèle en calculant l'erreur quadratique moyenne
        ...

    def __init__(self, sc, movies_set_path, ratings_set_path):
        # Méthode d'initialisation pour charger les ensembles de données et entraîner le modèle
        ...

```

- La classe `RecommendationEngine` est définie avec plusieurs méthodes qui implémentent les fonctionnalités nécessaires pour la recommandation de films.
- Chaque méthode est expliquée brièvement dans les commentaires, et vous pouvez les implémenter en utilisant le langage de programmation Python.

**Étape 3: Utilisation de la classe RecommendationEngine**

```python
# Création d'une instance de la classe RecommendationEngine
engine = RecommendationEngine(sc, "chemin_vers_ensemble_films", "chemin_vers_ensemble_evaluations")

# Exemple d'utilisation des méthodes de la classe RecommendationEngine
user_id = engine.create_user(None)
if engine.is_user_known(user_id):
    movie = engine.get_movie(None)
    ratings = engine.get_ratings_for_user(user_id)
    engine.add_ratings(user_id, ratings)
    prediction = engine.predict_rating(user_id, movie.movieId)
    recommendations = engine.recommend_for_user(user_id, 10)

```

- Pour utiliser la classe `RecommendationEngine`, vous devez créer une instance en passant le contexte Spark (`sc`) ainsi que les chemins vers les ensembles de données de films (`movies_set_path`) et d'évaluations (`ratings_set_path`).
- Vous pouvez ensuite utiliser les différentes méthodes de la classe pour créer de nouveaux utilisateurs, obtenir des films, ajouter de nouvelles évaluations, prédire des évaluations et obtenir des recommandations pour les utilisateurs.

