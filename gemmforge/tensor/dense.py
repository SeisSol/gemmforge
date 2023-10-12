from functools import reduce
from operator import mul
from typing import Union

from gemmforge.basic_types import DataFlowDirection


class DenseTensor:
  ADDRESSING = ["none", "strided", "pointer_based"]
  PTR_TYPES = {"none": "*",
               "strided": "*",
               "pointer_based": "**"}

  def __init__(self, dimensions, addressing, bbox=None):
    self.name = None
    self.dimensions = dimensions
    self.direction: Union[DataFlowDirection, None] = None
    self._real_dimensions = None
    self.bbox = bbox
    self.temporary = False

    if addressing in DenseTensor.ADDRESSING:
      self.addressing = addressing
      self.ptr_type = DenseTensor.PTR_TYPES[self.addressing]
    else:
      raise ValueError('Invalid matrix addressing. '
                       'Valid types: {}'.format(", ".join(DenseTensor.ADDRESSING)))

  def set_data_flow_direction(self, direction: DataFlowDirection):
    self.direction = direction

  def get_actual_num_dimensions(self):
    """
    if self._real_dimensions != None:
        return self._real_dimensions
    realDimensions = list()
    for i in range(len(self.dimensions)):
      realDimensions.append(self.bbox[i + len(self.dimensions)] - self.bbox[i])
    self._real_dimensions = realDimensions
    return self._real_dimensions
    """
    return self.dimensions

  def get_size(self):
    return self.get_actual_volume()

  def get_dimensions(self):
    return self.dimensions

  def get_padded_dimensions(self):
    return self.dimensions

  def get_actual_volume(self):
    realDimensions = self.get_actual_num_dimensions()
    actualVolume = reduce(mul, realDimensions)
    assert int(actualVolume) == actualVolume
    return int(actualVolume)

  def get_volume(self):
    volume = reduce(mul, self.dimensions)
    assert volume == int(volume)
    return int(volume)

  def get_real_volume(self):
    return self.get_volume()

  def get_total_size(self):
    return self.get_volume()

  def get_name(self):
    return self.name

  def get_accumulated_dimensions(self):
    cell_counts = [1]
    for dim in self.get_dimensions():
      cell_counts.append(dim * cell_counts[len(cell_counts) - 1])
    return cell_counts

  def get_elements_needed_for_dimensions_skip(self):
    skip_counts = [1]
    for dim in self.get_dimensions()[:-2]:
      skip_counts.append(dim * skip_counts[len(skip_counts) - 1])
    return skip_counts

  def get_offset_to_first_element(self):
    # partiallyReducedDimensions = [1]
    # for i in range(1, len(self.dimensions)):
    #  reducedOffset = partiallyReducedDimensions[i - 1] * self.dimensions[i - 1]
    #  assert reducedOffset == int(reducedOffset)
    #  partiallyReducedDimensions.append(int(reducedOffset))
    # print("PRD: ", partiallyReducedDimensions)

    # totalOffset  = 0
    # totalOffset += 1 * self.bbox[0]
    # for i in range(1, len(self.dimensions)):
    #  totalOffset += partiallyReducedDimensions[i - 1] * self.bbox[i - 1]
    return 0
    # print("TOTALOFFSET: ", totalOffset)

    # return int(totalOffset)

  def set_name(self, name):
    self.name = name

  def __str__(self):
    openbr = "{"
    closebr = "}"
    string = ""
    string += f"Tensor: {openbr} name = {self.name}, "
    string += f"addressing = {self.addressing}, "
    string += f"dimensions = {self.dimensions}, "
    string += f"dataflow direction = {self.direction}{closebr}"
    return string

  def __repr__(self):
    return self.__str__()

  def copy(self):
    clone = DenseTensor(self.dimensions, self.addressing,
                        self.bbox)
    clone.set_name(self.name)
    clone.set_data_flow_direction(self.direction)
    clone.temporary = self.temporary
    return clone