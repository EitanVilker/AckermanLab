def evaluate(predictions, answers):

    loss = 0
    successes = 0

    prediction_length = len(predictions)
    if prediction_length != len(answers):
        print("\n\nPrediction size not equal to answers size")
        return None

    for i in range(prediction_length):

        total_predictions += 1

        if predictions[i] == answers[i]:
            successes += 1
            loss += 0
        else:
            if testing_binary_classifier:
                loss += round(abs(predictions[i] - answers[i]))
            else:
                loss += round(abs(predictions[i] - answers[i]))

        return loss, successes, prediction_length