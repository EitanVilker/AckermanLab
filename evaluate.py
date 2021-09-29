
''' Function to assess the efficacy of the predictions. 
Currently returns total values instead of more standard metrics like MSE for ease of interpretation '''
def evaluate(predictions, answers):

    loss = 0
    successes = 0

    prediction_length = len(predictions)

    if prediction_length != len(answers):
        print("\n\nPrediction size not equal to answers size")
        return None

    for i in range(prediction_length):

        loss += abs(predictions[i] - answers[i])

        if round(predictions[i]) == answers[i]:
            successes += 1

    return loss, successes, prediction_length