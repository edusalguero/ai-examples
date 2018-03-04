"""
=====================================================
Calculate recommendations based on euclidean distance
=====================================================

In the example we use a dictionary of user movie reviews. Using euclidean distance you will get the distance between
the users and their interests similarity.

"""

import numpy as np

from prettytable import PrettyTable
from scipy.spatial.distance import euclidean


def distance(a, b):
    new_a = []
    new_b = []

    '''
    Find an remove zero values on arrays
    '''
    for i in range(0, len(a)):
        if a[i] != 0 and b[i] != 0:
            new_a.append(a[i])
            new_b.append(b[i])

    return round(euclidean(new_a, new_b), 2)


def similarity(a, b):
    return round(1 / (1 + distance(a, b)), 2)


def print_reviews(users_reviews):
    reviews_header = ['Reviews (User/Film)']
    users_count = len(users_reviews)
    films_count = len(users_reviews[0])
    for i in range(0, films_count):
        reviews_header.append(str(i + 1))
    p_reviews = PrettyTable(reviews_header)
    for i in range(0, users_count):
        user_row = [i + 1]
        for usersReview in users_reviews[i]:
            user_row.append(usersReview)
        p_reviews.add_row(user_row)
    print(p_reviews)


def print_distances(D):
    header = ['Distance']

    for index in range(0, len(D)):
        header.append(str(index + 1))

    table = PrettyTable(header)
    for d in distances:
        table.add_row(d)
    print(table)


def print_similarities(S):
    header = ['Similarity']

    for index in range(0, len(S)):
        header.append(str(index + 1))

    table = PrettyTable(header)
    for s in S:
        table.add_row(s)
    print(table)


usersReviews = np.array(
    [
        [0, 3, 1, 0, 4, 3, 5, 0],
        [4, 1, 3, 0, 5, 0, 0, 2],
        [2, 1, 0, 5, 0, 0, 0, 1],
        [3, 0, 2, 0, 0, 5, 0, 4]
    ])

distances = []
similarities = []

i = 0
for review in usersReviews:
    i = i + 1
    userDistances = [i]
    userSimilarities = [i]
    for review2 in usersReviews:
        userDistances.append(distance(review, review2))
        userSimilarities.append(similarity(review, review2))
    distances.append(userDistances)
    similarities.append(userSimilarities)

print_reviews(usersReviews)
print_distances(distances)
print_similarities(similarities)
