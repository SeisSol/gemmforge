from .abstract_gemmlike_generator_factory import AbstractGemmlikeGeneratorFactory
from .sycl_gemm_generator import SyclGemmGenerator

class SyclGemmGeneratorFactory(AbstractGemmlikeGeneratorFactory):

    def create(self, precision):
        return SyclGemmGenerator(self.arch, precision)
