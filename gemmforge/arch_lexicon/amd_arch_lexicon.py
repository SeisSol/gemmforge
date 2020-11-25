from .abstract_arch_lexicon import AbstractArchLexicon


class AmdArchLexicon(AbstractArchLexicon):

    def __init__(self):
        AbstractArchLexicon.__init__(self)
        self.threadIdx_y = "hipThreadIdx_y"
        self.threadIdx_x = "hipThreadIdx_x"
        self.blockIdx_x = "hipBlockIdx_x"
        self.blockDim_y = "hipBlockDim_y"

    def get_launch_code(self, func_name, grid, block, func_params):
        return "hipLaunchKernelGGL(kernel_{}, {}, {}, 0, 0, {})".format(func_name, grid, block, func_params)
