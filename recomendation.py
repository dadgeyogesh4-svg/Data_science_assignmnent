
"""
Created on Thu Nov 30 18:38:31 2023


import pandas as pd
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

data = {
    'UserID': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    'BookTitle': ['Book1', 'Book2', 'Book2', 'Book3', 'Book1', 'Book3', 'Book4', 'Book5', 'Book4', 'Book5'],
    'BookRating': [5, 4, 3, 5, 4, 2, 5, 1, 3, 4]
}

book_1 = pd.DataFrame(data)

user_item_matrix = book_1.pivot_table(index='UserID', columns='BookTitle', values='BookRating', fill_value=0)

scaler = MinMaxScaler()
scaled_user_item_matrix = scaler.fit_transform(user_item_matrix)

user_similarity = cosine_similarity(scaled_user_item_matrix, scaled_user_item_matrix)

def get_user_recommendations(user_id, user_similarity, user_item_matrix, data):
    similar_users = list(enumerate(user_similarity[user_id - 1]))
    sorted_similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)[1:]

    recommended_books = set()
    for similar_user, similarity_score in sorted_similar_users:
        unrated_books = user_item_matrix.loc[similar_user + 1][user_item_matrix.loc[user_id].values == 0]
        recommended_books.update(unrated_books[unrated_books > 0].index)

    recommended_books = list(recommended_books - set(user_item_matrix.columns[user_item_matrix.loc[user_id].values > 0]))

    return recommended_books[:5]  # You can change the number of recommendations as needed

user_id = 1
user_recommendations = get_user_recommendations(user_id, user_similarity, user_item_matrix, book_1)

print(f"Recommended books for User {user_id}:")
for i, book_title in enumerate(user_recommendations, start=1):
    print(f"{i}. {book_title}")
