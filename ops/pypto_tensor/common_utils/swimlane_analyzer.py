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
"""
GLM-4.5 MatMul AllReduce Module for performance

This module implements a fused matmul and all-reduce operation for large-scale distributed models.
It efficiently combines computation and communication, reducing memory overhead and accelerating training.

Main Functions:
    - matmul_allreduce: Main function for fused matmul and all-reduce computation
"""

import logging
import multiprocessing as mp
import os
import json
import csv
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone
import statistics


class SwimlaneAnalyzer:
    def __init__(self, output_dir: str, expected_total_time: Optional[float] = None):
        """
        初始化Swimlane分析器

        参数:
        output_dir: 输出目录路径
        expected_total_time: 预期总时间（微秒），在测试函数中设置
        """
        self.output_dir = output_dir
        self.performance_data = None
        self.expected_total_time = expected_total_time
        self._cached_swimlane_times = None

    @staticmethod
    def find_swimlane_files(rank_dir: str) -> List[str]:
        """在目录中查找 swimlane 文件"""
        swimlane_files = []
        for root, _, files in os.walk(rank_dir):
            for file in files:
                if file == "merged_swimlane.json":
                    swimlane_files.append(os.path.join(root, file))
        return swimlane_files

    @staticmethod
    def _calculate_total_time_from_data(performance_data: dict) -> float:
        """从性能数据计算总体执行时间（第一个任务开始到最后一个任务结束的时间跨度）"""
        if 'traceEvents' not in performance_data:
            raise ValueError("Performance data does not contain traceEvents")

        start_times = []
        end_times = []

        for event in performance_data['traceEvents']:
            if event.get('ph') == 'X':
                if 'fake' in event.get('name', '').lower():
                    continue

                start_time = event.get('ts', 0)
                duration = event.get('dur', 0)

                if duration <= 0:
                    continue

                end_time = start_time + duration
                start_times.append(start_time)
                end_times.append(end_time)

        if not start_times or not end_times:
            return 0.0

        overall_start = min(start_times)
        overall_end = max(end_times)
        total_time = overall_end - overall_start

        return total_time

    def find_all_rank_dirs(self) -> List[str]:
        """查找所有rank目录"""
        rank_dirs = []

        for item in os.listdir(self.output_dir):
            item_path = os.path.join(self.output_dir, item)
            if os.path.isdir(item_path):
                if 'rank' in item.lower():
                    rank_dirs.append(item_path)

        if not rank_dirs:
            for item in os.listdir(self.output_dir):
                item_path = os.path.join(self.output_dir, item)
                if os.path.isdir(item_path):
                    rank_dirs.append(item_path)

        return rank_dirs

    def find_recent_rank_dirs(self, world_size: int) -> List[str]:
        """查找最近的world_size个rank目录"""
        rank_dirs = self.find_all_rank_dirs()

        if not rank_dirs:
            return []

        rank_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        return rank_dirs[:world_size]

    def get_all_rank_times(self, world_size: int) -> List[Tuple[str, float]]:
        """
        获取所有rank文件的时间和路径

        返回:
        列表，每个元素是(文件路径, 时间)
        """
        # 如果已经有缓存，直接返回缓存结果
        if self._cached_swimlane_times is not None:
            return self._cached_swimlane_times

        rank_dirs = self.find_recent_rank_dirs(world_size)

        if not rank_dirs:
            raise FileNotFoundError(f"No rank directories found in {self.output_dir}")

        all_times = self._collect_swimlane_times(rank_dirs)

        if not all_times:
            raise ValueError("Could not find any valid swimlane file")

        # 缓存结果
        self._cached_swimlane_times = all_times

        return all_times

    def calculate_stats(self, world_size: int) -> Dict[str, float]:
        """计算统计信息：平均值、最小值、最大值"""
        all_times = self.get_all_rank_times(world_size)
        time_values = [time for _, time in all_times]

        if not time_values:
            return {
                'avg_time': 0.0,
                'min_time': 0.0,
                'max_time': 0.0,
                'num_ranks': 0
            }

        return {
            'avg_time': statistics.mean(time_values) if len(time_values) > 1 else time_values[0],
            'min_time': min(time_values),
            'max_time': max(time_values),
            'num_ranks': len(time_values)
        }

    def check_within_expected(self, world_size: int) -> bool:
        """检查最小执行时间是否在预期总时间内"""
        if self.expected_total_time is None:
            return True

        stats = self.calculate_stats(world_size)
        return stats['min_time'] <= self.expected_total_time

    def _collect_swimlane_times(self, rank_dirs: List[str]) -> List[Tuple[str, float]]:
        """收集所有 swimlane 文件的时间信息"""
        all_times = []
        for rank_dir in rank_dirs:
            swimlane_files = SwimlaneAnalyzer.find_swimlane_files(rank_dir)
            for file_path in swimlane_files:
                time_info = self._parse_swimlane_file(file_path)
                if time_info:
                    all_times.append(time_info)
        return all_times

    def _parse_swimlane_file(self, file_path: str) -> Optional[Tuple[str, float]]:
        """解析单个 swimlane 文件并返回时间信息"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            total_time = self._calculate_total_time_from_data(data)
            return (file_path, total_time)

        except (json.JSONDecodeError, OSError) as e:
            logging.warning(f"Failed to parse swimlane file {file_path}: {e}")
            return None