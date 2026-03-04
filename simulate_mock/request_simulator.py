import os
import asyncio
import json
import re
import requests
import logging
import time
import random
from collections import defaultdict
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# 环境变量
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL")
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL")

try:
    import httpx
except ImportError:
    raise SystemExit("需要安装 httpx: pip install httpx")

# 初始配置模板
CONFIG = {
    "url": f"{RAG_SERVICE_URL}/rag_generate",
    "collection_name": "act_test_kb",
    "timeout": 60.0,
    "retry_delay": 5.0,
    "retries": 5
}


def generate_question() -> str:
    """从LLM生成随机问题"""
    try:
        file_path = os.path.join(os.path.dirname(__file__), 'fortest.json')
        with open(file_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        
        response = requests.post(
            f"{LLM_SERVICE_URL}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        content = response.json()['choices'][0]['message']['content']
        question = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        logging.info(f"生成问题: {question[:50]}...")
        return question
    except Exception as e:
        logging.error(f"生成问题失败: {e}")
        return "默认测试问题"


async def send_request(client: httpx.AsyncClient, url: str, retries: int, retry_delay: float) -> dict:
    """发送单个请求并处理重试，每次尝试都生成新的query"""
    for attempt in range(retries + 1):
        try:
            # 每次尝试都生成新的问题
            question = generate_question()
            payload = {"collection_name": CONFIG["collection_name"], "query": question}
            
            start = time.perf_counter()
            response = await client.post(url, json=payload, timeout=CONFIG["timeout"])
            latency = time.perf_counter() - start
            
            return {
                "success": response.status_code < 400,
                "status": response.status_code,
                "latency": latency,
                "error": None if response.status_code < 400 else f"HTTP {response.status_code}"
            }
        except Exception as e:
            if attempt < retries:
                logging.warning(f"请求失败 (尝试 {attempt + 1}/{retries + 1}): {e}, 重新生成query并重试...")
                await asyncio.sleep(retry_delay)
            else:
                return {"success": False, "status": 0, "latency": 0, "error": str(e)}


async def run_test(rps: int, duration: float, concurrency: int):
    """执行负载测试"""
    url = CONFIG["url"]
    total_requests = int(rps * duration)
    interval = 1.0 / rps
    
    logging.info(f"开始测试 - RPS: {rps}, 持续: {duration:.1f}s, 并发: {concurrency}, 总请求: {total_requests}")
    
    stats = {"success": 0, "failure": 0, "latencies": [], "errors": defaultdict(int)}
    semaphore = asyncio.Semaphore(concurrency)
    
    async with httpx.AsyncClient() as client:
        start_time = time.perf_counter()
        
        async def execute_request(idx: int):
            # 按RPS控制发送时间
            target_time = start_time + idx * interval
            await asyncio.sleep(max(0, target_time - time.perf_counter()))
            
            # 发送请求（内部会为每次尝试生成新的query）
            async with semaphore:
                result = await send_request(client, url, CONFIG["retries"], CONFIG["retry_delay"])
            
            # 统计
            if result["success"]:
                stats["success"] += 1
                stats["latencies"].append(result["latency"])
            else:
                stats["failure"] += 1
                stats["errors"][result["error"]] += 1
        
        # 并发执行所有请求
        await asyncio.gather(*[execute_request(i) for i in range(total_requests)])
        elapsed = time.perf_counter() - start_time
    
    # 输出结果
    total = stats["success"] + stats["failure"]
    logging.info(f"测试完成 - 总数: {total}, 成功: {stats['success']}, 失败: {stats['failure']}, 耗时: {elapsed:.2f}s")
    
    if stats["latencies"]:
        latencies = sorted(stats["latencies"])
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        logging.info(f"延迟 - P50: {p50:.2f}s, P95: {p95:.2f}s, Max: {max(latencies):.2f}s")
    
    if stats["errors"]:
        logging.warning(f"错误统计: {dict(stats['errors'])}")


def run_random_test():
    """执行一次随机配置的测试"""
    rps = random.randint(1, 5)
    duration = random.uniform(10.0, 30.0)
    concurrency = random.randint(1, 5)
    
    try:
        asyncio.run(run_test(rps, duration, concurrency))
    except Exception as e:
        logging.error(f"测试失败: {e}")


def main():
    """主循环 - 每10分钟执行一次"""
    logging.info("模拟器启动，每10分钟执行一次随机测试")
    
    while True:
        run_random_test()
        logging.info("等待10分钟后执行下一次测试...")
        time.sleep(600)  # 10分钟


if __name__ == "__main__":
    main()

