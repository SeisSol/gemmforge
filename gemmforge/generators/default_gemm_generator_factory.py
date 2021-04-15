from .abstract_gemmlike_generator_factory import AbstractGemmlikeGeneratorFactory

from .gemm_generator import GemmGenerator


class DefaultGemmGeneratorFactory(AbstractGemmlikeGeneratorFactory):
    def create(self, precision):
        return GemmGenerator(self.arch, precision)
