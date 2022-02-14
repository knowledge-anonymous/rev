# coding=utf-8
# Copyright 2021 GBP Authors.
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

"""  """

from __future__ import print_function, absolute_import, division

import logging
from rich.logging import RichHandler

# FORMAT = "%(message)s"
# logging.basicConfig(
#     level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler(show_time=False)]
# )
#
# logger = logging.getLogger("gbp")
from gbp.utils import get_logger

logger = get_logger("gbp")
