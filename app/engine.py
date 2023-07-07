from pyspark.sql.types import *
from pyspark.sql.functions import explode, col
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
#from pyspark.sql.types import StructType, StructField, IntegerType, FloatType



class RecommendationEngine:
    #max_user_identifier = nb_users  # Attribut de classe pour stocker la valeur maximale de user_id
    #known_users = set()
    #best_movies_df = pd.DataFrame(...)  # Dataframe best_movies_df contenant les informations des meilleurs films
    #movies_df = pd.DataFrame(...)  # Dataframe movies_df contenant les informations de tous les films
    #ratings_df = pd.DataFrame(...)  # Dataframe ratings_df contenant les évaluations des utilisateurs
    #ratings_df = spark.createDataFrame(...) # Dataframe ratings_df contenant les évaluations existantes
    #model = ...  # Modèle utilisé pour les prédictions
    #model = ...  # Modèle utilisé pour les recommandations
    #movies_df = ...  # Dataframe contenant les détails des films
    #maxIter = ...
    #regParam = ...
    #training = ...

    def create_user(self, user_id=None):
        # Méthode pour créer un nouvel utilisateur
        # Vérifier si user_id est None
        if user_id is None:
            # Générer un nouvel identifiant en incrémentant max_user_identifier
            self.max_user_identifier += 1
            user_id = self.max_user_identifier
        else:
            # Mettre à jour max_user_identifier si user_id est supérieur à sa valeur actuelle
            if user_id > self.max_user_identifier:
                self.max_user_identifier = user_id

        # Retourner l'identifiant de l'utilisateur créé ou mis à jour
        return user_id


    def is_user_known(self, user_id):
        # Méthode pour vérifier si un utilisateur est connu
        if user_id is not None and user_id <= self.max_user_identifier:
            return True
        else:
            return False

    def get_movie(self, movie_id=None):
        # Méthode pour obtenir un film
        if movie_id is None:
            # Retourner un échantillon aléatoire d'un film à partir de best_movies_df
            random_movie = self.best_movies_df.sample(1)
            return random_movie[["movieId", "title"]]
        else:
            # Filtrer movies_df pour obtenir le film correspondant à movie_id
            movie = self.movies_df[self.movies_df["movieId"] == movie_id]
            return movie[["movieId", "title"]]


    def get_ratings_for_user(self, user_id):
        # Méthode pour obtenir les évaluations d'un utilisateur
        # Filtrer ratings_df pour obtenir les évaluations correspondantes à l'utilisateur
        user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]

        # Retourner un dataframe contenant les évaluations de l'utilisateur
        return user_ratings[["movieId", "userId", "rating"]]


    def add_ratings(self, user_id, ratings):
        # Méthode pour ajouter de nouvelles évaluations et re-entraîner le modèle
        #Méthode pour créer un nouveau dataframe new_ratings_df à partir de la liste de ratings 
        new_ratings_df =  spark.createDataFrame([(user_id, movie_id, rating) for movie_id, rating in ratings],["user_id", "movie_id", "rating"])
        #et l'ajoute au dataframe existant ratings_df en utilisant l'opération union() #mettre self. derrière ratings_df car c'est un paramètre
        self.ratings_df = self.ratings_df.union(new_ratings_df)
        # Diviser les données en ensembles d'entraînement (training) et de test (test) en utilisant randomSplit()
        training_data, test_data = self.ratings_df.randomSplit([0.8, 0.2])
        # Appeler la méthode privée __train_model() pour re-entraîner le modèle
        self.__train_model(training_data)


    def predict_rating(self, user_id, movie_id):
        # Méthode pour prédire une évaluation pour un utilisateur et un film donnés
        # Créer un dataframe rating_df à partir des données (user_id, movie_id)
        rating_df = spark.createDataFrame([(user_id, movie_id)], ["userId", "movieId"])

        # Transformer rating_df en utilisant le modèle pour obtenir les prédictions
        prediction_df = self.model.transform(rating_df)

        # Vérifier si le dataframe de prédiction est vide
        if prediction_df.isEmpty():
            return -1
        else:
            # Obtenir la valeur de prédiction
            prediction = prediction_df.select("prediction").collect()[0][0]
            return prediction


    def recommend_for_user(self, user_id, nb_movies):
        # Méthode pour obtenir les meilleures recommandations pour un utilisateur donné
        # Créer un dataframe user_df contenant l'identifiant de l'utilisateur
        user_df = spark.createDataFrame([(user_id,)], ["userId"])

        # Utiliser la méthode recommendForUserSubset() du modèle pour obtenir les recommandations pour cet utilisateur
        recommendations = self.model.recommendForUserSubset(user_df, nb_movies)

        # Joindre les recommandations avec le dataframe movies_df pour obtenir les détails des films recommandés
        recommended_movies_df = recommendations.join(self.movies_df, recommendations.movieId == self.movies_df.movieId)

        # Sélectionner les colonnes souhaitées du dataframe résultant
        result_df = recommended_movies_df.select("title", ...)

        return result_df


    def __train_model(self):
        # Méthode privée pour entraîner le modèle avec ALS
        # Créer une instance de l'algorithme ALS avec les paramètres maxIter et regParam
        als = ALS(maxIter=self.maxIter, regParam=self.regParam)

        # Entraîner le modèle en utilisant le dataframe training
        model = als.fit(self.training)

        # Appeler la méthode privée __evaluate() pour évaluer les performances du modèle
        self.__evaluate(model)

        # Retourner le modèle entraîné
        return model

    def __evaluate(self):
        # Méthode privée pour évaluer le modèle en calculant l'erreur quadratique moyenne
        # Utiliser le modèle pour prédire les évaluations sur le dataframe test
        predictions = self.model.transform(self.test)

        # Créer un objet RegressionEvaluator pour calculer le RMSE
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

        # Calculer le RMSE en comparant les prédictions avec les vraies évaluations
        rmse = evaluator.evaluate(predictions)

        # Stocker la valeur de RMSE dans la variable rmse de la classe
        self.rmse = rmse

        # Afficher la valeur de RMSE à l'écran
        print(f"RMSE: {rmse}")


    def __init__(self, sc, movies_set_path, ratings_set_path):
        # Méthode d'initialisation pour charger les ensembles de données et entraîner le modèle
        # Initialiser le contexte Spark
        spark = SparkSession(sc)

        # Charger les données des ensembles de films et d'évaluations depuis les fichiers CSV
        movies_data = spark.read.csv(movies_set_path, header=True)
        ratings_data = spark.read.csv(ratings_set_path, header=True)

        # Définir le schéma des données
        movies_schema = StructType([
            StructField("movieId", IntegerType(), True),
            StructField("title", StringType(), True),
            StructField("genres", StringType(), True),
            StructField("timestamp", TimestampType(), True)
            # Ajouter d'autres colonnes du schéma des films si nécessaire
        ])

        ratings_schema = StructType([
            StructField("userId", IntegerType(), True),
            StructField("movieId", IntegerType(), True),
            StructField("rating", FloatType(), True),
            # Ajouter d'autres colonnes du schéma des évaluations si nécessaire
        ])

        # Appliquer le schéma aux données des ensembles de films et d'évaluations
        movies_df = spark.createDataFrame(movies_data.rdd, movies_schema)
        ratings_df = spark.createDataFrame(ratings_data.rdd, ratings_schema)

        # Effectuer d'autres opérations de traitement des données si nécessaire

        # Entraîner le modèle en utilisant la méthode privée __train_model()
        self.__train_model()