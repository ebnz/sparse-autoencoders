import torch
from torch.utils.data import Dataset
import os


class TokenizedDataset(Dataset):
    def __init__(self, path, dtype=torch.float32):
        """
        Represents a Dataset of Tokenized texts
        :type path: str
        :param path: Path, where the Dataset is located
        :param dtype: Datatype, used in the Dataset
        """
        self.path = path
        self.files = os.listdir(self.path)
        self.i = 0

        self.current_file_id = 0
        self.file = torch.load(os.path.join(self.path, self.files[0]))
        self.files = os.listdir(self.path)

        self.NUM_FILES = len(self.files)
        self.NUM_SAMPLES_PER_FILE = len(self.file)

        self.dtype = dtype

    def __len__(self):
        """
        Returns the Number of Items in the Dataset.
        :rtype: int
        :return: Length of Dataset
        """
        return self.NUM_FILES * self.NUM_SAMPLES_PER_FILE

    def __iter__(self):
        """
        Returns an Iterator of this Dataset.
        :return: Iterator
        """
        return self

    def __getitem__(self, index):
        """
        Get an Item from the Dataset by Index.
        :type index: int
        :rtype: torch.Tensor
        :param index: Index of the Item to return
        :return: Dataset Item
        """
        if index >= self.NUM_FILES * self.NUM_SAMPLES_PER_FILE:
            raise Exception(f"End of Dataset reached. Requested item {index} but length of Dataset is "
                            f"{self.NUM_FILES * self.NUM_SAMPLES_PER_FILE}")

        new_file_id = index // self.NUM_SAMPLES_PER_FILE
        sub_index = index % self.NUM_SAMPLES_PER_FILE

        if new_file_id != self.current_file_id:
            self.file = torch.load(os.path.join(self.path, self.files[new_file_id]))
            self.current_file_id = new_file_id

        return self.file[sub_index].to(self.dtype)

    def __next__(self):
        """
        Return next Element of Iterator.
        :rtype: torch.Tensor
        :return: Dataset Item
        """
        rv = self[self.i]
        self.i += 1
        return rv


class TokenizedDatasetPreload(TokenizedDataset):
    def __init__(self, path, partial_preload=None, dtype=torch.float32):
        """
        Represents a Dataset of Tokenized texts. This Dataset can be partially loaded using partial_preload.
        :type partial_preload: float
        :param partial_preload: Ratio of the Dataset to Preload. Float between 0 and 1
        """
        super().__init__(path, dtype=dtype)

        self.partial_preload = partial_preload

        if self.partial_preload is not None:
            self.NUM_FILES = int(self.partial_preload * len(self.files))
            self.files = self.files[:self.NUM_FILES]
        
        self.data = [torch.load(os.path.join(self.path, file)) for file in self.files]

    def __getitem__(self, index):
        if index >= self.NUM_FILES * self.NUM_SAMPLES_PER_FILE:
            raise Exception(f"End of Dataset reached. Requested item {index} but length of Dataset is "
                            f"{self.NUM_FILES * self.NUM_SAMPLES_PER_FILE}")

        new_file_id = index // self.NUM_SAMPLES_PER_FILE
        sub_index = index % self.NUM_SAMPLES_PER_FILE

        return self.data[new_file_id][sub_index].to(self.dtype)
