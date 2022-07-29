from operator import itemgetter

from gemmforge.exceptions import GenerationError
from gemmforge.basic_types import DataFlowDirection
from typing import Union


from gemmforge.matrix.matrix import Matrix

#Here is the functioning class of SparseMatrix
class MockMatrix(Matrix):

  def __init__(self,  num_rows, num_cols, addressing , spp  , values = None , bbox=None):
    super().__init__(num_rows, num_cols, addressing, bbox)
    self.name = None
    self.num_rows = num_rows
    self.num_cols = num_cols
    self.direction: Union[DataFlowDirection, None] = None
    self.values = values
    self.spp = spp
    self.bbox=bbox
    for i in self.spp:
      if (int(i[0]) > self.num_rows) or (int(i[1]) > self.num_cols) or (int(i[0]) >= self.bbox[2]) or (int(i[1]) >= self.bbox[3]) or (int(i[1]) < 0):
         raise ValueError('unvalid sparsity pattern ')
    for i in self.spp:
      if (int(i[0]) > self.num_rows) or (int(i[1]) > self.num_cols) or (int(i[0]) >= self.bbox[2]) or (int(i[1]) >= self.bbox[3]) or (int(i[1]) < 0):
        raise ValueError('unvalid sparsity pattern ')
    # check if the values and the sparsity pattern have the same length
    if( values != None):
        if (len(self.values) != len(self.spp)):
            raise ValueError('sparsity pattern list is not compatible with the value list')


  def __str__(self):
   string = super().__str__()
   string += "spp= {}\n".format(self.spp)
   return string
