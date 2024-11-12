import json
import pandas as pd
import pyarrow as pa
import os
from tqdm import tqdm
from collections import defaultdict, Counter
from .glossary import normalize_word
import random

def get_score(occurences):
    if occurences == 0:
        return 0.0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1.0

def path2rest(entry, root, ans2label, label2ans):
    image_path = os.path.join(root, entry["image"])
    with open(image_path, "rb") as fp:
        binary = fp.read()

    question = entry["question"]
    answer = entry["answer"]
    question_id = entry["question_id"]
    image_id = int(image_path.split("/")[-1].split("-")[0])  # Assuming image_id is derived from the filename

    questions = [question]  # Wrap in a list to maintain schema
    answers = [[answer]]  # Wrap in a list of lists to maintain schema

    # Normalize and label the answer
    normalized_answer = normalize_word(answer)
    if normalized_answer in ans2label:
        answer_label = [ans2label[normalized_answer]]
        answer_score = [1.0]  # Assuming a single score
    else:
        answer_label = []
        answer_score = []

    answer_labels = [answer_label]  # Wrap in a list of lists to maintain schema
    answer_scores = [answer_score]  # Wrap in a list of lists to maintain schema
    split = "val"  # Assuming all entries are in the validation split

    return [binary, questions, answers, answer_labels, answer_scores, image_id, [question_id], split]

def make_arrow(root: str, json_file: str, dataset_root: str):
    """
    Function to create an Arrow file from the given JSON file

    Parameters:
    - root (str): Root directory containing the images
    - json_file (str): Path to the JSON file containing the data
    - dataset_root (str): Root directory to save the Arrow file
    """
    with open(json_file, "r") as fp:
        data = json.load(fp)

    # Collect all answers to create label mappings
    all_answers = [normalize_word(entry["answer"]) for entry in data]
    counter = {k: v for k, v in Counter(all_answers).items() if v >= 9}
    ans2label = {k: i for i, k in enumerate(counter.keys())}
    label2ans = list(counter.keys())

    bs = [path2rest(entry, root, ans2label, label2ans) for entry in tqdm(data)]

    dataframe = pd.DataFrame(
        bs,
        columns=[
            "image",
            "questions",
            "answers",
            "answer_labels",
            "answer_scores",
            "image_id",
            "question_id",
            "split",
        ],
    )

    # table = pa.Table.from_pandas(dataframe)

    # Sample 1 percent of the dataframe
    sample_size = int(len(dataframe) * 0.01)
    sampled_dataframe = dataframe.sample(n=sample_size, random_state=42)

    table = pa.Table.from_pandas(sampled_dataframe)

    os.makedirs(dataset_root, exist_ok=True)
    with pa.OSFile(f"{dataset_root}/vqa_vlue_test.arrow", "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)