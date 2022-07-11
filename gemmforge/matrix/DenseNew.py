from gemmforge.exceptions import GenerationError
from gemmforge.basic_types import DataFlowDirection


from gemmforge.matrix.matrix import Matrix
from typing import Union


class DenseMatrix(Matrix):

  def __init__(self,  num_rows, num_cols, addressing , bbox=None):
    super().__init__(num_rows, num_cols, addressing, bbox)
    self.name = None
    self.num_rows = num_rows
    self.num_cols = num_cols
    self.direction: Union[DataFlowDirection, None] = None
    self.bbox = bbox





  def __str__(self):
   string = super().__str__(self)
   return string
