import json
import os
import pandas as pd
import pyarrow as pa
from tqdm import tqdm
from glob import glob

def path2rest(path, iid2captions):
    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[path]
    split = "val"  # Adding split information
    return [binary, captions, path, split]

def make_arrow(root: str, json_file: str, dataset_root: str):

    """
    Function to create an arrow file from the OOD dataset

    Parameters:
    - root (str): path to the root directory of the OOD dataset containing the images
    - json_file (str): path to the json file containing the captions
    - dataset_root (str): path to the directory where the arrow file will be saved
    """
    with open(json_file, "r") as fp:
        data = json.load(fp)

    iid2captions = {os.path.join(root, item["image"]): item["caption"] for item in data}

    paths = list(iid2captions.keys())
    caption_paths = [path for path in paths if os.path.exists(path)]

    if len(paths) == len(caption_paths):
        print("all images have caption annotations")
    else:
        print("not all images have caption annotations")
    print(len(paths), len(caption_paths))

    bs = [path2rest(path, iid2captions) for path in tqdm(caption_paths)]

    dataframe = pd.DataFrame(bs, columns=["image", "caption", "image_id", "split"])

    # table = pa.Table.from_pandas(dataframe)

    # Sample 1 percent of the dataframe
    sample_size = int(len(dataframe) * 0.01)
    sampled_dataframe = dataframe.sample(n=sample_size, random_state=42)

    table = pa.Table.from_pandas(sampled_dataframe)

    os.makedirs(dataset_root, exist_ok=True)
    with pa.OSFile(f"{dataset_root}/itr_vlue_test.arrow", "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)
