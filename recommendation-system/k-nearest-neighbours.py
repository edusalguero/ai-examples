"""
=====================================================
Calculate recommendations based on k-nearest neighbours
=====================================================

In the example we use a dictionary of user movie reviews. Using SVD you
will predict the user interests in a film.

"""
import surprise.dataset

dataPath = '../data/ml-100k/u.data'

columns = 'user item rating timestamp'

reader = surprise.dataset.Reader(line_format=columns, sep='\t')

data = surprise.dataset.Dataset.load_from_file(dataPath, reader=reader)

# Extract the training subset and train kNN with it
train = data.build_full_trainset()

method = surprise.KNNBasic()
method.fit(train)

# Estimate a value that already exists
user = str(196)
item = str(242)
prediction = method.predict(user, item, r_ui=3)
print('Estimated = ', prediction.est, ' real= ', prediction.r_ui)

# Estimate a valuation that does not exist in the data (the user / element pair is not found)
user = str(196)
item = str(291)

prediction = method.predict(user, item)

print('Estimated =', prediction.est)

# kNN with different k and using Pearson's correlation as # measure of similarity
sim_options = {'name': 'pearson'}
method2 = surprise.KNNWithMeans(k=20, sim_options=sim_options)
method2.fit(train)
# method2 is evaluated using RMSE and MAE as quality measures
results = surprise.model_selection.cross_validate(method2, data, measures=['RMSE', 'MAE'])
surprise.print_perf(results)

# Predict
prediction = method2.predict(user, item)
print('Estimated =', prediction.est)
