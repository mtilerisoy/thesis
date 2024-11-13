import json
import pandas as pd
import pyarrow as pa
import os

from tqdm import tqdm


def process(root, entry):
    texts = [entry["sentence"]]  # Ensure texts is a list
    labels = [entry["label"]]    # Ensure labels is a list
    img_paths = entry["images"]

    img0_path = os.path.join(root, img_paths[0])
    img1_path = os.path.join(root, img_paths[1])

    with open(img0_path, "rb") as fp:
        img0 = fp.read()
    with open(img1_path, "rb") as fp:
        img1 = fp.read()

    identifier = os.path.basename(img0_path).split('.')[0]

    return [img0, img1, texts, labels, identifier]


def make_arrow(root, dataset_root):
    with open(f"{root}/nlvr2_vlue_test.json") as f:
        data = json.load(f)

    bs = [process(root, entry) for entry in tqdm(data)]

    dataframe = pd.DataFrame(
        bs, columns=["image_0", "image_1", "questions", "answers", "image_id"],
    )

    table = pa.Table.from_pandas(dataframe)

    os.makedirs(dataset_root, exist_ok=True)
    with pa.OSFile(f"{dataset_root}/nlvr2_vlue_test.arrow", "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)
