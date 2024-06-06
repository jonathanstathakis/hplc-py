from hplc_py.definitions import PRECISION


class Precision:
    
    __precision = PRECISION
    def __init__(self):
        self.test = "hi"

    @property
    def _precision(self):
        return self.__precision

    @_precision.getter
    def _precision(self):
        return self.__precision