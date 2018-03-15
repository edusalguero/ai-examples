"""
=====================================================
Calculate recommendations based on Pearson correlation coefficient
=====================================================

In the example we use a dictionary of user movie reviews. Using pearson correlation coefficient  you
will get the similarity between their interests.

"""

import numpy as np
import math

from prettytable import PrettyTable
from scipy.spatial.distance import correlation


def similarity(a, b):
    # Get common elements and remove 0 values (no review)
    commons_a = []
    commons_b = []
    for j in range(0, len(a)):
        if a[j] != 0 and b[j] != 0:
            commons_a.append(a[j])

    for j in range(0, len(b)):
        if b[j] != 0 and a[j] != 0:
            commons_b.append(b[j])

    commons_count = len(commons_a)

    # If there are no common elements, return zero; otherwise
    # compute the coefficient
    if commons_count == 0:
        return 0

    pearson_correlation = correlation(commons_a, commons_b)
    # If divisor is zero
    if math.isnan(pearson_correlation):
        return 0

    return round(1 - pearson_correlation, 2)


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


def print_similarities(S):
    header = ['Similarity']

    for index in range(0, len(S)):
        header.append(str(index + 1))

    table = PrettyTable(header)
    for s in S:
        table.add_row(s)
    print(table)


"""
The users reviews matrix
"""
usersReviews = np.array(
    [
        [0, 3, 1, 0, 4, 3, 5, 0],
        [4, 1, 3, 0, 5, 0, 0, 2],
        [2, 1, 0, 5, 0, 0, 0, 1],
        [3, 0, 2, 0, 0, 5, 0, 4]
    ])

similarities = []

i = 0
for review in usersReviews:
    i = i + 1
    userSimilarities = [i]
    for review2 in usersReviews:
        userSimilarities.append(similarity(review, review2))
    similarities.append(userSimilarities)

print_reviews(usersReviews)
print_similarities(similarities)
