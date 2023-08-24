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
    realDimensions = list()
    for i in range(len(self.dimensions)):
      realDimensions.append(self.bbox[i + len(self.dimensions)] - self.bbox[i])
    return realDimensions

  def get_actual_volume(self):
    realDimensions = self.get_actual_num_dimensions()
    actualVolume = reduce(mul, realDimensions)
    return actualVolume

  def get_real_volume(self):
    actualVolume = reduce(mul, self.dimensions)
    return actualVolume

  def get_offset_to_first_element(self):
    partiallyReducedDimensions = [1]
    for i in range(1, len(self.dimensions)):
      reducedOffset = partiallyReducedDimensions[i - 1] * self.dimensions[i - 1]
      partiallyReducedDimensions.append(reducedOffset)
    print("PRD: ", partiallyReducedDimensions)

    totalOffset  = 0
    totalOffset += [1 * self.bbox[0]]
    for i in range(1, len(self.bbox)):
      totalOffset += partiallyReducedDimensions[i - 1] * self.bbox[i - 1]
    print("TOTALOFFSET: ", totalOffset)

    return totalOffset

  def set_name(self, name):
    self.name = name

  def __str__(self):
    string = "num. rows = {}\n".format(self.num_rows)
    string += "num. columns = {}\n".format(self.num_cols)
    string += "bounding box = {}\n".format(self.bbox)
    string += "addressing = {}\n".format(self.addressing)
    string += "dimensions = {}\n".format(self.dimensions)
    return string
