from operator import itemgetter

from gemmforge.exceptions import GenerationError


from gemmforge.matrix.matrix import Matrix


class SparseMatrix(Matrix):

  def __init__(self, spp, values , bbox=None):
    super().__init__(self)
    self.spp = spp
    self.values = values

    if bbox is not None:
      self.bbox = bbox

      # check whether bbox were given correctly
      coords = [coord for coord in self.bbox]
    #   if (3 < self.get_actual_num_rows()) or (
    #      len(self.spp) < self.get_actual_num_cols()):
    #     raise GenerationError('Matrix size {}x{} is '
    #                           'smaller than bbox {}')
    #   if (3 < self.bbox[2]) or (len(self.spp)< self.bbox[3]):
    #     raise GenerationError('Bbox {} is '
    #                           'outside of Matrix {}x{}')
    # else:
    #   self.bbox = (0, 0, 3, len(self.values))

      if (self.num_rows < self.get_actual_num_rows()) or (
        self.num_cols < self.get_actual_num_cols()):
        raise GenerationError('Matrix size {}x{} is '
                              'smaller than bbox {}'.format(self.num_rows,
                                                            self.num_cols,
                                                            coords))
      if (self.num_rows < self.bbox[2]) or (self.num_cols < self.bbox[3]):
        raise GenerationError('Bbox {} is '
                              'outside of Matrix {}x{}'.format(coords,
                                                               self.num_rows,
                                                               self.num_cols))
      else:
        rows_spp = max(spp, key=itemgetter(0))[0]
        cols_spp = max(spp, key=itemgetter(1))[1]
      self.bbox = (0, 0, rows_spp, cols_spp)

    # todo: the implementation should handle pythonList , Python tuples and PythonNumPy arrays
    # check whether sparsity pattern is correct or not
      for i in self.spp:
        if (i[0] >= self.num_rows) or (i[1] >= self.num_cols) or (i[0] < 0) or (i[1] < 0):
          raise ValueError('unvalid sparsity pattern ')

    # check if the values and the sparsity pattern have the same length
    if (len(self.values) != len(self.spp)):
      raise ValueError('sparsity pattern list is not compatible with the value list')

  def get_actual_num_rows(self):
    return self.bbox[2] - self.bbox[0]

  def get_actual_num_cols(self):
    return self.bbox[3] - self.bbox[1]

  def get_actual_volume(self):
    return self.get_actual_num_rows() * self.get_actual_num_cols()

  def get_real_volume(self):
    return self.num_rows * self.num_cols

  def get_offset_to_first_element(self):
    return self.num_rows * self.bbox[1] + self.bbox[0]

  def __str__(self):
    string = super().__str__(self)
    string += "num. actual rows = {}\n".format(self.get_actual_num_rows())
    string += "num. actual cols = {}\n".format(self.get_actual_num_cols())
    return string
