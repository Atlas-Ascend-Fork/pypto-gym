#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


import logging


def create_logger(name):
    # Configure logger for the module
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    LOG_FORMATTER = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='[%Y-%m-%d %H:%M:%S]'
    )
    LOG_HANDLER = logging.StreamHandler()
    LOG_HANDLER.setFormatter(LOG_FORMATTER)
    logger.handlers.clear()
    logger.addHandler(LOG_HANDLER)

    return logger
