#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    :   2025/06/17 19:21:00
@Author  :   wangjiakang
@File    :   __init__.py
'''


from .registry import get_reward_manager_cls, register  # noqa: I001
from .batch import BatchRewardManager
from .dapo import DAPORewardManager
from .naive import NaiveRewardManager
from .prime import PrimeRewardManager
from .wizard import WizardRewardManager

# Note(haibin.lin): no need to include all reward managers here in case of complicated dependencies
__all__ = ["BatchRewardManager", "DAPORewardManager", "NaiveRewardManager", "PrimeRewardManager", "WizardRewardManager", "register", "get_reward_manager_cls"]
