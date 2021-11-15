


class ModelSVCWrapper:
    def __init__(self):
        pass

    def predict(self, models_svc, X_train):

        result = []
        Y_predict_bug = models_svc["bug"].predict_proba(X_train)
        Y_predict_feature = models_svc["feature"].predict_proba(X_train)
        Y_predict_rating = models_svc["rating"].predict_proba(X_train)
        Y_predict_userexperience = models_svc["userexperience"].predict_proba(X_train)

        X_train_array = X_train.to_numpy()
        for index, val in enumerate(X_train_array):
            predict_review_type = ""
            Y_predict_max = 0

            Y_predict_bug_now = Y_predict_bug[index][1]
            if (Y_predict_bug_now > Y_predict_max):
                Y_predict_max = Y_predict_bug_now
                predict_review_type = 0 # bug

            Y_predict_feature_now = Y_predict_feature[index][1]
            if (Y_predict_feature_now > Y_predict_max):
                Y_predict_max = Y_predict_feature_now
                predict_review_type = 1 # feature

            Y_predict_rating_now = Y_predict_rating[index][1]
            if (Y_predict_rating_now > Y_predict_max):
                Y_predict_max = Y_predict_rating_now
                predict_review_type = 2 # rating

            Y_predict_userexperience_now = Y_predict_userexperience[index][1]
            if (Y_predict_userexperience_now > Y_predict_max):
                Y_predict_max = Y_predict_userexperience_now
                predict_review_type = 3 # userexperience

            # predict_review_type_prob_feature = models_svc["feature"].predict_proba(X)
            result.append(predict_review_type)
            # predict_review_type_prob_userexperience = models_svc["userexperience"].predict_proba(X)

        return result