import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

df = pd.read_csv("movies.csv")

def build_model():
    global vectorizer, similarity_matrix
    vectorizer = TfidfVectorizer()
    genre_matrix = vectorizer.fit_transform(df["genre"])
    similarity_matrix = cosine_similarity(genre_matrix)

build_model()

def add_movie(movie_name):
    global df
    genre = input("Enter genre/description: ").strip()

    if genre == "":
        print("Invalid genre.")
        return

    new_row = {"title": movie_name.title(), "genre": genre}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv("movies.csv", index=False)
    build_model()

    print("Movie added successfully.")

def recommend_movie(movie_name, top_n=5):
    movie_name = movie_name.lower()
    titles_lower = df["title"].str.lower().values

    if movie_name not in titles_lower:
        matches = difflib.get_close_matches(movie_name, titles_lower, n=1, cutoff=0.6)

        if matches:
            movie_name = matches[0]
        else:
            print("Movie not found.")
            choice = input("Do you want to add this movie? (yes/no): ")

            if choice.lower() == "yes":
                add_movie(movie_name)
                movie_name = movie_name.lower()
            else:
                return

    index = df[df["title"].str.lower() == movie_name].index[0]

    scores = list(enumerate(similarity_matrix[index]))
    sorted_movies = sorted(scores, key=lambda x: x[1], reverse=True)

    print("\nRecommended Movies:\n")

    count = 0
    for movie in sorted_movies[1:]:
        print(df.iloc[movie[0]]["title"])
        count += 1
        if count == top_n:
            break


print("🎬 Movie Recommendation System")

print("\nAvailable Movies:")
for movie in sorted(df["title"]):
    print("-", movie)

print("\nType 'exit' to stop")

while True:

    user_input = input("\nEnter Movie Name: ")

    if user_input.lower() == "exit":
        print("Program Closed.")
        break

    try:
        top_n = int(input("Number of recommendations (1-5): "))

        if top_n < 1 or top_n > 5:
            print("Enter number between 1 and 5")
            continue

    except:
        print("Invalid number")
        continue

    recommend_movie(user_input, top_n)