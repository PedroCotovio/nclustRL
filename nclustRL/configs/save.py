
import pickle
import os

class TrainerConfig:

    def __init__(self, filename, path) -> None:
        
        self._filename = filename
        self._path = path

        try:
            self.obj = self.load()

        except:
            self.obj = None

    @property
    def path(self):
        return os.path.join(self._path, self._filename)

    def load(self):

        with open(self.path + '.pickle', 'rb') as handle:
            b = pickle.load(handle)

        return b

    def save(self, obj):

        with open(self.path + '.pickle', 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return None


    