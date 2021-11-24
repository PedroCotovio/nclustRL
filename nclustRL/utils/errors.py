
class BaseError(Exception):

    def __init__(self, val, message):
        super(self.__init__(message.format(val)))


class EnvError(BaseError):

    def __init__(self, val):
        message = '{} environment does not exist in nclustEnv.'
        super(self.__init__(val, message))


class TrainerError(BaseError):

    def __init__(self, val):
        message = '{} trainer does not exist in RlLib.'
        super(self.__init__(val, message))
