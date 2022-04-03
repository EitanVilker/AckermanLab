from sklearn.model_selection import permutation_test_score

''' Function to assess the efficacy of the predictions. 
Currently returns total values instead of more standard metrics like MSE for ease of interpretation '''
def evaluate(predictions, answers, nominal=False):

    loss = 0
    successes = 0
    prediction_length = len(predictions)
    if prediction_length != len(answers):
        print("\n\nPrediction size not equal to answers size")
        return None

    if not nominal:
        for i in range(prediction_length):
            loss += abs(predictions[i] - answers[i])
            if round(predictions[i]) == answers[i]:
                successes += 1

        return loss, successes, prediction_length
    else:
        for i in range(prediction_length):
            if predictions[i] == answers[i]:
                successes += 1
        return loss, successes, prediction_length

def permutation_test(estimator, attributes, classifier):
    print("\nScore, permutation scores, p-value: ")
    print(permutation_test_score(estimator, attributes, classifier))