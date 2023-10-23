from pycaret.classification import *

exp_clf = setup(data=merged_df, target=LABEL, session_id=123, normalize=True)

candidate_algorithms = ['dt', 'rf', 'knn', 'nb', 'xgboost']

best_model = None
best_model_score = 0

for algorithm in candidate_algorithms:
    model = create_model(algorithm)
    tuned_model = tune_model(model)

    evaluation = evaluate_model(tuned_model)

    if evaluation['F1'] > best_model_score:
        best_model = tuned_model
        best_model_score = evaluation['F1']

print("Best Model:")
print(best_model)
