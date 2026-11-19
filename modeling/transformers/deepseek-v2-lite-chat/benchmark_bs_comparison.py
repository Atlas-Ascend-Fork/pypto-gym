# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""
Batch Size性能对比测试脚本
对比Baseline ACLGraph和PyPTO ACLGraph在不同BS组合下的性能
"""

import logging
import subprocess
import json
import time
import re
import os
import sys

os.chdir(os.environ.get("MODEL_PATH", "/path/to/models/DeepSeek-V2-Lite-Chat"))

test_configs = [
    # BS=1, 不同输出长度（影响KV cache长度）
    {'bs': 1, 'output_length': 10, 'prompt': '你好'},
    {'bs': 1, 'output_length': 20, 'prompt': '你好'},
    {'bs': 1, 'output_length': 50, 'prompt': '你好，请介绍一下你自己'},
    {'bs': 1, 'output_length': 100, 'prompt': '请详细介绍一下人工智能的发展历程'},
    
    # 不同batch size（需要修改脚本支持多batch）
    # 当前脚本不支持多batch，先用单batch不同输出长度测试
]



def run_test(use_kv_fusion, output_length, prompt, device=7):
    """
    运行单次测试
    
    Args:
        use_kv_fusion: bool - 是否使用PyPTO KV融合
        output_length: int - 输出token数量
        prompt: str - 输入提示
        device: int - NPU设备ID
    
    Returns:
        dict: {'throughput': float, 'inference_time': float, 'tokens': int}
    """
    cmd = [
        'python3', 
        'scripts/ask_DeepSeek-V2-Lite-Chat.py',
        '--prompt', prompt,
        '--device', str(device),
        '--use_acl_graph',
        '--output_length', str(output_length),
    ]
    
    if use_kv_fusion:
        cmd.append('--use_kv_fusion')
    
    mode_name = "PyPTO ACLGraph" if use_kv_fusion else "Baseline ACLGraph"
    logging.info(f"\n{'='*60}")
    logging.info(f"{'='*60}")
    logging.info(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    elapsed = time.time() - start_time
    
    # 解析输出
    output = result.stdout + result.stderr
    if result.returncode != 0:
        logging.error("Inference process exited with code %s", result.returncode)
        logging.info("Output snippet:\n%s", output[-500:])
        return {
            'success': False,
            'mode': mode_name,
            'error': f'Inference process exited with code {result.returncode}',
        }
    
    # 提取性能指标（更宽松的正则）
    throughput_match = re.search(r'吞吐量.*?([\d.]+).*?tokens/s', output)
    inference_match = re.search(r'推理耗时.*?([\d.]+).*?s', output)
    tokens_match = re.search(r'生成token数.*?(\d+)', output)
    
    if throughput_match and inference_match and tokens_match:
        throughput = float(throughput_match.group(1))
        inference_time = float(inference_match.group(1))
        tokens = int(tokens_match.group(1))
        
        logging.info(f"✓ Success: {tokens} tokens, {inference_time:.2f}s, {throughput:.1f} tokens/s")
        
        return {
            'throughput': throughput,
            'inference_time': inference_time,
            'tokens': tokens,
            'total_time': elapsed,
            'success': True,
            'mode': mode_name,
        }
    else:
        logging.error(f"✗ Failed: unable to parse performance metrics")
        logging.info(f"Output snippet:\n{output[-500:]}")
        return {
            'success': False,
            'mode': mode_name,


            'error': 'Failed to parse metrics',
        }


def main():
    logging.info("\n" + "="*80)
    logging.info("Batch Size Performance Comparison Test")
    logging.info("Comparing Baseline ACLGraph vs PyPTO ACLGraph")
    logging.info("="*80)
    
    results = []
    
    for config in test_configs:
        bs = config['bs']
        output_length = config['output_length']
        prompt = config['prompt']
        
        logging.info(f"\n{'#'*80}")
        logging.info(f"Config: BS={bs}, OutputLen={output_length}, Prompt='{prompt[:30]}...'")
        logging.info(f"{'#'*80}")
        
        # 测试Baseline ACLGraph
        baseline_result = run_test(
            use_kv_fusion=False,
            output_length=output_length,
            prompt=prompt
        )
        
        # 等待NPU冷却
        time.sleep(5)
        
        # 测试PyPTO ACLGraph
        pypto_result = run_test(
            use_kv_fusion=True,
            output_length=output_length,
            prompt=prompt
        )
        
        # 记录结果
        results.append({
            'config': config,
            'baseline': baseline_result,
            'pypto': pypto_result,
        })
        
        # 等待NPU冷却
        time.sleep(10)
    
    # 生成对比报告
    logging.info("\n" + "="*80)
    logging.info("Performance Comparison Summary")
    logging.info("="*80)
    
    logging.info(f"\n{'BS':>4} {'OutputLen':>10} {'Baseline':>12} {'PyPTO':>12} {'Speedup':>10} {'Status':>10}")
    logging.info("-" * 80)
    
    for r in results:
        config = r['config']
        baseline = r['baseline']
        pypto = r['pypto']
        
        if baseline['success'] and pypto['success']:
            baseline_tps = baseline['throughput']
            pypto_tps = pypto['throughput']
            
            if baseline_tps > 0:
                speedup = (pypto_tps / baseline_tps - 1) * 100
                speedup_str = f"{speedup:+.1f}%"
            else:
                speedup_str = "N/A"
            
            if abs(speedup) < 5:
                status = "~Same"
            elif speedup > 0:
                status = "✓ PyPTO Faster"
            else:
                status = "Warning Baseline Faster"
            
            logging.info(f"{config['bs']:>4} {config['output_length']:>10} "
                  f"{baseline_tps:>12.1f} {pypto_tps:>12.1f} "
                  f"{speedup_str:>10} {status:>10}")
        else:
            logging.info(f"{config['bs']:>4} {config['output_length']:>10} "
                  f"{'FAILED':>12} {'FAILED':>12} {'N/A':>10} {'ERROR':>10}")
    
    # 保存结果到JSON
    result_file = 'benchmark_bs_results.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logging.info(f"\n✓ Results saved to: {result_file}")
    
    # 详细分析
    logging.info("\n" + "="*80)
    logging.info("Detailed Analysis")
    logging.info("="*80)
    
    for r in results:
        config = r['config']
        baseline = r['baseline']
        pypto = r['pypto']
        
        logging.info(f"\nConfig: BS={config['bs']}, OutputLen={config['output_length']}")
        
        if baseline['success']:
            logging.info(f"  Baseline ACLGraph:")
            logging.info(f"    - Throughput: {baseline['throughput']:.2f} tokens/s")
            logging.info(f"    - Inference time: {baseline['inference_time']:.2f}s")
            logging.info(f"    - Generated tokens: {baseline['tokens']}")
        
        if pypto['success']:
            logging.info(f"  PyPTO ACLGraph:")
            logging.info(f"    - Throughput: {pypto['throughput']:.2f} tokens/s")
            logging.info(f"    - Inference time: {pypto['inference_time']:.2f}s")
            logging.info(f"    - Generated tokens: {pypto['tokens']}")
            
            if baseline['success'] and baseline['throughput'] > 0:
                speedup = (pypto['throughput'] / baseline['throughput'] - 1) * 100
                logging.info(f"  Performance diff: {speedup:+.2f}%")

if __name__ == '__main__':
    main()
