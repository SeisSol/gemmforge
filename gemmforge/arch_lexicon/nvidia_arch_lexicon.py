from .abstract_arch_lexicon import AbstractArchLexicon


class NvidiaArchLexicon(AbstractArchLexicon):

    def __init__(self):
        AbstractArchLexicon.__init__(self)
        self.threadIdx_y = "threadIdx.y"
        self.threadIdx_x = "threadIdx.x"
        self.blockIdx_x = "blockIdx.x"
        self.blockDim_y = "blockDim.y"

    def get_launch_code(self, func_name, grid, block, func_params):
        return "kernel_{}<<<{},{}>>>({})".format(func_name, grid, block, func_params)

