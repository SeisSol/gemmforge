from abc import ABC, abstractmethod


class AbstractGemmlikeGeneratorFactory(ABC):

    def __init__(self, arch):
        self.arch = arch

    @abstractmethod
    def create(self, precision):
        pass
