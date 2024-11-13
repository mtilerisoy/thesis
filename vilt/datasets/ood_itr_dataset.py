from .base_dataset import BaseDataset


class OODITRDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]

        if split == "train":
            # names = ["f30k_caption_karpathy_train", "f30k_caption_karpathy_val"]
            names = ["itr_vlue_test"]
        elif split == "val":
            names = ["itr_vlue_test"]
        elif split == "test":
            names = ["itr_vlue_test"]

        # Pass the names to load the datasets from arrow files
        # Pass the text_column_name to specify the column name for the text input (changes depending on the task)
        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

    def __getitem__(self, index):
        return self.get_suite(index)
