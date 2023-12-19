
 # Copyright (c) 2023, salesforce.com, inc.
 # All rights reserved.
 # SPDX-License-Identifier: BSD-3-Clause
 # For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

from tqdm import tqdm
import pickle
import numpy as np
from PIL import Image

data, labels = pickle.load(open('modelnet40_test_1024pts.dat', 'rb'))