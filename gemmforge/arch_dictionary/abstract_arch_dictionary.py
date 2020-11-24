from abc import ABC, abstractmethod


class AbstractArchDictionary(ABC):
    """
    You can use this abstract class to add a dictionary for any manufacturer for variables like e.g. threadIdx.x for
    CUDA that are used by the generators and loaders
    """

    def __init__(self):
        self.threadIdx_x = None
        self.threadIdx_y = None
        self.blockDim_y = None
        self.blockIdx_x = None

    def get_tid_counter(self):
        return "(" + self.threadIdx_y + " + " + self.blockDim_y + " * " + self.blockIdx_x + ")"

    def get_thread_idx_x(self):
        return self.threadIdx_x

    def get_thread_idx_y(self):
        return self.threadIdx_y

    def get_blockdim_y(self):
        return self.blockDim_y

    def get_block_idx_x(self):
        return self.blockIdx_x

    @abstractmethod
    def get_launch_code(self, func_name, grid, block, func_params):
        pass
