import json
import argparse

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
        qid = pred["question_id"]
        original_answer = next((item["answer"] for item in original_answers if item["question_id"] == qid), None)

        if original_answer is not None and pred["answer"] == original_answer:
            correct += 1
        total += 1

    print(f"Correct: {correct}, Total: {total}")
    accuracy = correct / total if total > 0 else 0
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VQA predictions.")
    
    parser.add_argument('model', type=str, default='vilt', help='model name: vilt or meter')
    parser.add_argument('pred_file', type=str, default='ViLT_original_predictions.json', help='Filename of the prediction json. Can eb found under results')
    args = parser.parse_args()

    pred_file = args.pred_file
    model_type = args.model

    if "vilt" in model_type:
        ans_file = "ViLT_original_answers.json"
    
    elif "meter" in model_type:
        ans_file = "METER_original_answers.json"
    
    else:
        raise ValueError("Model type not supported")

    pred_file = "result/" + pred_file + ".json"
    ans_file = "result/vqa/" + ans_file

    predictions = load_predictions(pred_file)
    original_answers = load_original_answers(ans_file)
    accuracy = calculate_accuracy(predictions, original_answers)

    print(f"Accuracy: {accuracy * 100:.2f}%")