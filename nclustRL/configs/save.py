
import pickle
import os

class TrainerConfig:

    def __init__(self, filename, path) -> None:
        
        self._filename = filename
        self._path = path
        self.obj = None

        try:
            self.load()

        except:
            pass

    @property
    def path(self):
        return os.path.join(self._path, self._filename)

    def load(self):

        with open(self.path + '.pickle', 'rb') as handle:
            b = pickle.load(handle)

        self.obj = b

        return self.obj

    def save(self, obj):

        with open(self.path + '.pickle', 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.obj = obj

        return None


    