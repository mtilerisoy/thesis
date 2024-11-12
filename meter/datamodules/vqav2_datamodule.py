from ..datasets import VQAv2Dataset
from .datamodule_base import BaseDataModule
from collections import defaultdict


class VQAv2DataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return VQAv2Dataset

    @property
    def dataset_name(self):
        return "vqa"

    def setup(self, stage):
        super().setup(stage)

        train_answers = self.train_dataset.table["answers"].to_pandas().tolist()
        val_answers = self.val_dataset.table["answers"].to_pandas().tolist()
        train_labels = self.train_dataset.table["answer_labels"].to_pandas().tolist()
        val_labels = self.val_dataset.table["answer_labels"].to_pandas().tolist()

        print(f"Train answers: {train_answers[:5]}")
        print("############################################")
        print(f"Train labels: {train_labels[:5]}")
        print("############################################")
        print(f"Val answers: {val_answers[:5]}")
        print("############################################")
        print(f"Val labels: {val_labels[:5]}")
        print("############################################")

        all_answers = [c for c in train_answers + val_answers if c is not None]
        all_answers = [l for lll in all_answers for ll in lll for l in ll]
        all_labels = [c for c in train_labels + val_labels if c is not None]
        all_labels = [l for lll in all_labels for ll in lll for l in ll]

        self.answer2id = {k: v for k, v in zip(all_answers, all_labels)}
        
        sorted_a2i = sorted(self.answer2id.items(), key=lambda x: x[1])
        self.num_class = max(self.answer2id.values()) + 1
        

        self.id2answer = defaultdict(lambda: "unknown")
        for k, v in sorted_a2i:
            self.id2answer[v] = k
        
        print(f"ANSWER2ID: {self.answer2id}")
        print("############################################")
        print(f"Sorted answer2id: {sorted_a2i}")
        print("############################################")
        print(f"ID2ANSWER: {self.id2answer}")
        print("############################################")
        print(f"Some ID2ANSWER conversions: {self.id2answer[0]}, {self.id2answer[5]}, {self.id2answer[10]}")
        print("############################################")
        
        raise SystemExit("Terminating the program")
