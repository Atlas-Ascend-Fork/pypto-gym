#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Atomic file replacement and exclusive access to one campaign's CLI state."""
import fcntl
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


class CampaignBusy(RuntimeError):
    pass


class CampaignUnavailable(RuntimeError):
    pass


def atomic_write(path, text):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=directory, prefix=".tmp-",
                                         suffix=".json", delete=False, encoding="utf-8") as handle:
            temporary = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.remove(temporary)


@contextmanager
def campaign_lock(op_dir):
    if not op_dir:
        yield  # Stateless normalization has no campaign state to protect.
        return
    directory = Path(op_dir) / "optimization"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        handle = (directory / ".campaign.lock").open("a")
    except OSError as exc:
        raise CampaignUnavailable(str(exc)) from exc
    # Keep the inode after release: unlinking a lock can admit two owners.
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignBusy(str(op_dir)) from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        handle.close()
