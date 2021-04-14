from .abstract_arch_lexic import AbstractArchLexic


class SyclArchLexic(AbstractArchLexic):

    def __init__(self):
        AbstractArchLexic.__init__(self)
        self.threadIdx_y = "item.get_local_id(1)"
        self.threadIdx_x = "item.get_local_id(0)"
        self.threadIdx_z = "item.get_local_id(2)"
        self.blockIdx_x = "item.get_group().get_id(0)"
        self.blockDim_y = "item.get_group().get_id(1)"
        self.blockDim_z = "item.get_group().get_id(2)"
        self.stream_name = "cl::sycl::queue"

    def get_launch_code(self, func_name, grid, block, stream, func_params):
        return f"kernel_{func_name}({stream}, {grid}, {block}, {func_params})";

