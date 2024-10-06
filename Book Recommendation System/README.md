## BOOK RECOMMENDATION SYSTEM 

### Overview

This project implements a Book Recommender System designed to help users discover new books based on their preferences and interactions. 
The system primarily uses two recommendation strategies: popularity-based filtering and collaborative filtering.

The goal of the system is to assist readers in exploring books they might enjoy by leveraging user behavior and book popularity.

### Features

- **Popularity-based Filtering:** Recommends books that are popular across all users. This method ranks books based on factors like the
number of ratings and average rating, making it ideal for new users or when there is insufficient data about user preferences. <br> <br>
- **Collaborative Filtering:** Suggests books based on user interactions. This technique finds similar users or books by analyzing user-book 
interactions (such as ratings) and suggests books that users with similar tastes have liked. Used **Cosine Similarity** to measure the similarity
between the books based on their ratings by the user.

### Dataset
The system uses the following datasets:

- **Books:** Contains details such as book titles, authors, and publication details.
- **Users:** Contains anonymized user information.
- **Ratings:** Contains the ratings provided by users for different books. <br> <br>

Downloaded the dataset from the provided github link : https://github.com/mujtabaali02/Book-Recommendation-System/tree/master


___