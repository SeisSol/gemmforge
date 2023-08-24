from gemmforge.exceptions import GenerationError
from gemmforge.basic_types import DataFlowDirection
from typing import Union
from functools import reduce
from operator import mul

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

    if bbox is not None:
      self.bbox = bbox

      # check whether bbox were given correctly
      coords = [coord for coord in self.bbox]
      for i in range(len(self.dimensions)):
        if self.dimensions[i] < coords[len(self.dimensions) + i]:
          raise GenerationError(f"Tensor size and bbox are compatible, dimensions: {self.dimensions}, bbox: {coords}",)
    else:
      tmpBbox = []
      for i in range(len(self.dimensions)):
        tmpBbox.append(0)
      tmpBbox += self.dimensions
      self.bbox = tmpBbox

    if addressing in DenseTensor.ADDRESSING:
      self.addressing = addressing
      self.ptr_type = DenseTensor.PTR_TYPES[self.addressing]
    else:
      raise ValueError('Invalid matrix addressing. '
                       'Valid types: {}'.format(", ".join(DenseTensor.ADDRESSING)))

  def set_data_flow_direction(self, direction: DataFlowDirection):
    self.direction = direction

  def get_actual_num_dimensions(self):
    if self._real_dimensions != None:
        return self._real_dimensions
    realDimensions = list()
    for i in range(len(self.dimensions)):
      realDimensions.append(self.bbox[i + len(self.dimensions)] - self.bbox[i])
    self._real_dimensions = realDimensions
    return self._real_dimensions

  def get_size(self):
    return self.get_actual_volume()

  def get_dimensions(self):
    return self.get_actual_num_dimensions()

  def get_padded_dimensions(self):
    return self.dimensions

  def get_actual_volume(self):
    realDimensions = self.get_actual_num_dimensions()
    actualVolume = reduce(mul, realDimensions)
    assert int(actualVolume) == actualVolume
    return int(actualVolume)

  def get_volume(self):
    actualVolume = reduce(mul, self.dimensions)
    assert actualVolume == int(actualVolume)
    return int(actualVolume)

  def get_offset_to_first_element(self):
    partiallyReducedDimensions = [1]
    for i in range(1, len(self.dimensions)):
      reducedOffset = partiallyReducedDimensions[i - 1] * self.dimensions[i - 1]
      assert reducedOffset == int(reducedOffset)
      partiallyReducedDimensions.append(int(reducedOffset))
    print("PRD: ", partiallyReducedDimensions)

    totalOffset  = 0
    totalOffset += 1 * self.bbox[0]
    for i in range(1, len(self.dimensions)):
      totalOffset += partiallyReducedDimensions[i - 1] * self.bbox[i - 1]
    print("TOTALOFFSET: ", totalOffset)

    return int(totalOffset)

  def set_name(self, name):
    self.name = name

  def __str__(self):
    string = ""
    string += "bounding box = {}\n".format(self.bbox)
    string += "addressing = {}\n".format(self.addressing)
    string += "dimensions = {}\n".format(self.dimensions)
    return string
