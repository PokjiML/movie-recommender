import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, ratings_path, movies_path):
        self.ratings_path = ratings_path
        self.movies_path = movies_path
        self.ratings = None
        self.movies = None
        self.user_movie_df = None

    def load_data(self):
        try:
            self.ratings = pd.read_csv(self.ratings_path)
            self.movies = pd.read_csv(self.movies_path)
        except FileNotFoundError as e:
            print(f"Couldn't find the dataset")
            raise
        
    def preprocess_data(self):
        try:
            self.ratings.drop('timestamp', axis=1, inplace=True)
        except Exception as e:
            print("Column already removed")

    def calculate_user_based_matrix(self):
        self.user_movie_df = pd.pivot_table(self.ratings, index='userId', 
                                            columns='movieId', values='rating')


class RecommenderSystem:
    def __init__(self, ratings, movies, user_movie_df, user_id):
        self.ratings = ratings
        self.movies = movies
        self.user_movie_df = user_movie_df
        self.user_id = user_id

        self.neighbors_ids = {}
        self.user_similarities = {}
        self.predicted_ratings = {}

        

    def get_15_user_movies(self):
        merge_df = pd.merge(self.ratings, self.movies, on='movieId')
        watched_movies = merge_df[merge_df['userId'] == self.user_id]
        print(watched_movies[['title', 'genres']].set_index('title').head(15))

        

    def calculate_user_similarity(self, K=5):
        """ Return sorted list of most
            similar user Ids """
        

        # Use only overlapping reviews between two users
        for iter_id in self.user_movie_df.index.drop(self.user_id):
            # Get the data
            two_users_df = self.user_movie_df.loc[[self.user_id, iter_id]]
            overlapping_df = two_users_df.dropna(axis=1, how='any')

            # Calculate only if two users rated at least 21 mutual movies
            if overlapping_df.shape[1] < 20:
                continue

            # Calculate the norms of both vectors for denominator
            vector_norm = overlapping_df.apply(lambda row: np.sqrt(np.sum(row**2)), axis=1)
            norms_product = vector_norm.iloc[0] * vector_norm.iloc[1]

            # Calculate the dot product between the vectors
            dot_product = np.dot(overlapping_df.iloc[0], overlapping_df.iloc[1])

            # calculate the similarity measure
            similarity = dot_product / norms_product

            # add the iter_id person similarity to dict
            self.user_similarities[iter_id] = similarity

        # Sort dict based on similarities
        sorted_sims = dict(sorted(self.user_similarities.items(), 
                   key=lambda item: item[1], reverse=True))
        
        # Return K most similar neighbors Ids
        self.neighbors_ids = list(sorted_sims.keys())[:K]

       
    

    def calculate_movie_predictions(self):

        # Calculate the avarage of a user
        avg_user_rating = self.user_movie_df.loc[user_id].mean()


        for movie_id in self.user_movie_df.columns:

            sum_nominator = 0
            sum_denominator = 0
            
            for n_id in self.neighbors_ids:

                # Get the neighbor similarity
                neighbor_sim = self.user_similarities[n_id]

                # Calculate the avarage rating of a neighbor
                avg_neighbor_rating = self.user_movie_df.loc[n_id].mean()

                # Calculate the rating of a given movie from a given user
                iter_movie_rating = self.user_movie_df.loc[n_id, movie_id]

                # Skip if the movie hasn't been rated
                if pd.isna(iter_movie_rating):
                    continue

                # Difference between a given rating and the mean
                rating_diff = iter_movie_rating - avg_neighbor_rating


                sum_nominator += self.user_similarities[n_id] * rating_diff
                sum_denominator += np.abs(neighbor_sim) # Check if absolute is better


            # If no common movies return avarage rating
            if sum_denominator == 0:
                pred_rating = avg_user_rating
            else:
                pred_rating = avg_user_rating + (sum_nominator/sum_denominator)

            self.predicted_ratings[movie_id] = pred_rating

    
    def get_recommended_movies(self, N=5):

        # Sort movies
        sorted_pred_rankings = list(sorted(self.predicted_ratings.items(), 
                                           key = lambda item: item[1], reverse=True))
        
        print(f"\nThe movies recommended for user {self.user_id}:\n")

        for movie, corr in sorted_pred_rankings[:5]:
            print(self.movies[self.movies.movieId == movie].iloc[0].title)



if __name__ == "__main__":

    user_id = int(input('Select user Id: '))
    
    
    data_loader = DataLoader("ml-latest-small/ratings.csv", 
                             "ml-latest-small/movies.csv")
    
    data_loader.load_data()
    data_loader.preprocess_data()
    data_loader.calculate_user_based_matrix()

    # Check if userId is in the dataset
    if user_id not in data_loader.ratings['userId'].unique():
        print("User ID could not be found. Exiting")
        exit()

    recommender = RecommenderSystem(data_loader.ratings, data_loader.movies,
                                    data_loader.user_movie_df, user_id)
    
    recommender.get_15_user_movies()

    recommender.calculate_user_similarity(K=10)
    recommender.calculate_movie_predictions()

    recommender.get_recommended_movies()

