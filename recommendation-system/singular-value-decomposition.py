"""
=====================================================
Estimate recommendations based on SVD
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
method = surprise.SVD()
method.fit(train)

# Valuate system
results = surprise.model_selection.cross_validate(method, data, measures=['RMSE', 'MAE'])
surprise.print_perf(results)

# Params adjustment
param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005]}
search = surprise.model_selection.GridSearchCV(surprise.SVD, param_grid, measures=['RMSE'], cv=None, refit=False,
                                               return_train_measures=True)
search.fit(data)
# Show better result and parameters with which it has been achieved
print(search.best_score['rmse'])
print(search.best_params['rmse'])

# Predict
user = str(196)
item = str(291)
prediction = method.predict(user, item)
print('Estimated =', prediction.est)
