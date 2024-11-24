import json

def load_predictions(pred_file):
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    return predictions

def load_original_answers(ans_file):
    with open(ans_file, 'r') as f:
        original_answers = json.load(f)
    return original_answers

def calculate_accuracy(predictions, original_answers):
    correct = 0
    total = 0

    for pred in predictions:
        # if pred["answer"] == "Unknown Answer":
        #     continue

        qid = pred["question_id"]
        original_answer = next((item["answer"] for item in original_answers if item["question_id"] == qid), None)

        if original_answer is not None and pred["answer"] == original_answer:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy

if __name__ == "__main__":
    pred_file = "result/vqa_submit_vilt_vqa.json"
    ans_file = "result/vqa/ViLT_original_answers.json"

    predictions = load_predictions(pred_file)
    original_answers = load_original_answers(ans_file)
    accuracy = calculate_accuracy(predictions, original_answers)

    print(f"Accuracy: {accuracy * 100:.2f}%")