from .abstract_generator import AbstractGenerator as Generator
from .gemm_generator import GemmGenerator


class SyclGemmGenerator(GemmGenerator):
    def declare_shared_memory_inline(self, name, precision, size):
        return None

    def kernel_definition(self, file, kernel_bounds):
        localmem = "cl::sycl::accessor<{}, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> Scratch ({}, cgh);".format(self.precision, self._get_total_shared_mem_size())
        return file.SyclKernel(self.base_name, self._get_func_params(), kernel_bounds, localmem)

    def sync_threads(self):
        return "item.barrier()"

    def kernel_range_object(self):
        return "cl::sycl::range<3>"

    def get_stream_via_pointer(self, file):
        file.Expression("cl::sycl::queue defaultQueue {cl::sycl::host_selector{}}")
        if_stream_exists = f'({Generator.STREAM_PTR_STR} != nullptr)'
        stream_obj = f'static_cast<{self.arch_lexic.get_stream_name()} *>({Generator.STREAM_PTR_STR})'
        file(f'{self.arch_lexic.get_stream_name()} *stream = {if_stream_exists} ? {stream_obj} : &defaultQueue;')

    def synch_stream_pointer(self, stream):
        return f"{stream}->wait()"

    def check_error(self):
        return None