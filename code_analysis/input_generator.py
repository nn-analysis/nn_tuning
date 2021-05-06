from abc import ABC


class InputGenerator(ABC):

    def generate(self, shape: tuple):
        raise NotImplementedError
