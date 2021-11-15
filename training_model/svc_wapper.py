


class ModelSVCWrapper:
    def __init__(self):
        pass

    def predict(self, models_svc, X):
        predict_review_type = ""
        predict_review_type_prob = 0

        predict_review_type_prob_bug = models_svc["bug"].predict_proba(X)
        predict_review_type_prob_feature = models_svc["feature"].predict_proba(X)
        predict_review_type_prob_rating = models_svc["rating"].predict_proba(X)
        predict_review_type_prob_userexperience = models_svc["userexperience"].predict_proba(X)

        pass