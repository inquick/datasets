# coding=utf-8
# Copyright 2019 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""UC Merced: Small remote sensing dataset for land use classification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """\
@InProceedings{Nilsback08,
   author = "Yang, Yi and Newsam, Shawn",
   title = "Bag-Of-Visual-Words and Spatial Extensions for Land-Use Classification",
   booktitle = "ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (ACM GIS)",
   year = "2010",
}"""

_DESCRIPTION = """\
This is a 21 class land use remote sensing image dataset, with 100 images per
class. The images were manually extracted from large images from the USGS
National Map Urban Area Imagery collection for various urban areas around the
country. The pixel resolution of this public domain imagery is 0.3 m.
Each image measures 256x256 pixels."""

_LABELS = [
    "agricultural",
    "airplane",
    "baseballdiamond",
    "beach",
    "buildings",
    "chaparral",
    "denseresidential",
    "forest",
    "freeway",
    "golfcourse",
    "harbor",
    "intersection",
    "mediumresidential",
    "mobilehomepark",
    "overpass",
    "parkinglot",
    "river",
    "runway",
    "sparseresidential",
    "storagetanks",
    "tenniscourt",
]


class UcMerced(tfds.core.GeneratorBasedBuilder):
  """Small 21 class remote sensing land use classification dataset."""

  VERSION = tfds.core.Version("0.0.1")

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "image": tfds.features.Image(),
            "label": tfds.features.ClassLabel(names=_LABELS),
        }),
        supervised_keys=("image", "label"),
        urls=["http://weegee.vision.ucmerced.edu/datasets/landuse.html"],
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    if not tf.io.gfile.exists(dl_manager.manual_dir):
      raise AssertionError("You must download the dataset files manually and "
                           "place them in {}".format(dl_manager.manual_dir))
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            num_shards=1,
            gen_kwargs={"path": dl_manager.manual_dir},
        ),
    ]

  def _generate_examples(self, path):
    """Yields examples."""
    for subdir in tf.io.gfile.glob(path + "/train/*"):
      label = os.path.basename(subdir)
      for filename in tf.io.gfile.glob(subdir + "/*.tif"):
        image = _load_tif(filename)
        yield {"image": image, "label": label}


def _load_tif(path):
  with tf.io.gfile.GFile(path, "rb") as fp:
    image = tfds.core.lazy_imports.PIL_Image.open(fp)
  return np.array(image)
