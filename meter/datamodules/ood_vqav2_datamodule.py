from ..datasets import OODVQAv2Dataset
from .datamodule_base import BaseDataModule
from collections import defaultdict
import numpy as np


class OODVQAv2DataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return OODVQAv2Dataset

    @property
    def dataset_name(self):
        return "ood_vqa"

    def flatten_list(self, nested_list):
        """
        Flatten a nested list up to 3 levels deep. The code is based on VQAv2DataModule setup method.
        
        Args:
            nested_list (list): A nested list.
            
        Returns:
            list: A flattened list.
        """
        if isinstance(nested_list[0][0], list):
            # 3-layer nested list
            return [l for lll in nested_list for ll in lll for l in ll]
        else:
            # 2-layer nested list
            return [l for ll in nested_list for l in ll]
    
    def setup(self, stage):
        super().setup(stage)
    
        train_answers = self.train_dataset.table["answers"].to_pandas().tolist()
        val_answers = self.val_dataset.table["answers"].to_pandas().tolist()
        train_labels = self.train_dataset.table["answer_labels"].to_pandas().tolist()
        val_labels = self.val_dataset.table["answer_labels"].to_pandas().tolist()
    
        # print(f"Train answers: {train_answers[:5]}")
        # print("############################################")
        # print(f"Train labels: {train_labels[:5]}")
        # print("############################################")
        # print(f"Val answers: {val_answers[:5]}")
        # print("############################################")
        # print(f"Val labels: {val_labels[:5]}")
        # print("############################################")
    
        all_answers = [c for c in train_answers + val_answers if c is not None]
        all_answers = self.flatten_list(all_answers)
        all_labels = [c for c in train_labels + val_labels if c is not None]
        all_labels = self.flatten_list(all_labels)
    
        # Ensure all elements are hashable by converting numpy arrays to tuples
        all_answers = [tuple(a) if isinstance(a, np.ndarray) else a for a in all_answers]
        all_labels = [tuple(l) if isinstance(l, np.ndarray) else l for l in all_labels]
    
        self.answer2id = {k: v for k, v in zip(all_answers, all_labels)}
        sorted_a2i = sorted(self.answer2id.items(), key=lambda x: x[1])
        
        # Ensure all values are integers before finding the max
        all_label_values = [item for sublist in all_labels for item in sublist]
        self.num_class = max(all_label_values) + 1
    
        self.id2answer = defaultdict(lambda: "Unknown Answer") # Default value for unknown answers
        for k, v in sorted_a2i:
            self.id2answer[v] = k
    
        # print(f"ANSWER2ID: {self.answer2id}")
        # print("############################################")
        # print(f"Sorted answer2id: {sorted_a2i}")
        # print("############################################")
        # print(f"ID2ANSWER: {self.id2answer}")
        # print("############################################")
        # print(f"Some ID2ANSWER conversions: {self.id2answer[0]}, {self.id2answer[5]}, {self.id2answer[10]}")
        # print("############################################")
    
        # raise SystemExit("Terminating the program")