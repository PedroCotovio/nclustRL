
class BaseError(Exception):

    def __init__(self, val, message):
        super().__init__(message.format(val))


class EnvError(BaseError):

    def __init__(self, val):
        message = '{} environment does not exist in nclustEnv.'
        super().__init__(val, message)


class TrainerError(BaseError):

    def __init__(self, val):
        message = '{} trainer does not exist in RlLib.'
        super().__init__(val, message)


class DatasetError(BaseError):

    def __init__(self, val):
        message = '{} dataset must be an instance of SynteticDataset.'
        super().__init__(val, message)
