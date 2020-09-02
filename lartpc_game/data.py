import scipy.sparse
import numpy as np
from pathlib import Path
from collections import  OrderedDict


class LartpcData:

    def __init__(self, source_list: list, target_list: list):
        self.source_list = source_list
        self.target_list = target_list
        self.index = 0
        self.length = len(source_list)

    @staticmethod
    def from_path(data_filepath):
        data_filepath = Path(data_filepath)
        source_files_list = [x for x in data_filepath.iterdir() if 'image' in x.name]
        source_files_range = [ int(x.name[len('image'):].split('.')[0]) for x in source_files_list]
        source_dict = OrderedDict(sorted(zip(source_files_range,source_files_list), key=lambda x:x[0]))
        source_list = [ path for index, path in  source_dict.items()]
        target_files_list = [x for x in data_filepath.iterdir() if 'label' in x.name]
        target_files_range = [ int(x.name[len('label'):].split('.')[0]) for x in target_files_list]
        target_dict = OrderedDict(sorted(zip(target_files_range,target_files_list), key=lambda x:x[0]))
        target_list = [ path for index, path in target_dict.items()]
        return LartpcData(source_list, target_list)

    def __len__(self):
        return self.length

    def _read_array(self, npz_path) -> np.ndarray:
        matrix = scipy.sparse.load_npz(npz_path)
        matrix_dense = matrix.todense()
        np_arr = np.asarray(matrix_dense)
        return np_arr

    def __getitem__(self, item):
        s_path, s_target = self.source_list[item], self.target_list[item]
        source, target = self._read_array(s_path), self._read_array(s_target)
        return source, target

    def get_range(self, min, max):
        return LartpcData(self.source_list[min:max], self.target_list[min:max])

    def random(self):
        return self[np.random.randint(0, len(self))]

    def current(self):
        return self[self.index]

    def __next__(self):
        if self.index > self.length:
            raise StopIteration
        else:
            self.index += 1
            return self[self.index-1]

