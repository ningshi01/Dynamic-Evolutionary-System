import os
import asyncio
import json
import re
import requests
import logging
import statistics
# import schedule
import random
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import httpx
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This script requires httpx. Install it with 'pip install httpx'.") from exc


"""
Environment-driven request simulator with built-in defaults.

All parameters have internal defaults defined below and can be optionally overridden
by a JSON config file. If SIM_CONFIG_FILE is set in the environment, it will be used;
otherwise the script looks for ./request_simulator.config.json next to this file.

Built-in defaults (override via config file if needed):
    url: "http://192.168.5.65:31890/rag_generate"
    rps: 1.0
    duration: 60.0
    timeout: 10.0
    method: "POST"
    concurrency: 1
    retries: 0
    retry_delay: 1.0
    expected_status: null
    log_level: "INFO"
    verbose: false
    json: true
    headers: {}
    payload: null
    payload_file: null

Config JSON schema (all optional):
{
    "url": "http://192.168.5.65:31890/rag_generate",
    "rps": 10,
    "duration": 120,
    "timeout": 15,
    "method": "POST",
    "concurrency": 3,
    "retries": 1,
    "retry_delay": 0.5,
    "expected_status": 200,
    "log_level": "INFO",
    "verbose": false,
    "json": true,
    "headers": {"Content-Type": "application/json"},
    "payload": {
        "collection_name": "act_test_kb",
        "query": "测试一下"
    },
    "payload_file": "./payloads.json"
}
"""

def generate_random_payload() -> dict:
    # 构建文件的绝对路径
    file_path = os.path.join(os.path.dirname(__file__), 'fortest.json')

    # 从文件加载JSON数据
    with open(file_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    
    response = requests.post("http://192.168.5.65:30566/v1/chat/completions", headers={
        "Content-Type": "application/json"
    }, json=payload)
    # print("response:", response.json())
    content = response.json()['choices'][0]['message']['content']
    print("question:", re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip())
    return {
        "collection_name": "act_test_kb",
        "query": re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    }

# =====================
# Built-in default config
# =====================
DEFAULTS: Dict[str, Any] = {
    "url":            "http://192.168.5.65:31890/rag_generate",  # 目标服务的URL地址，RAG生成服务的端点
    "rps":            2.0,                                       # 每秒请求数，控制请求频率
    "duration":       20.0,                                       # 测试持续时间（秒）
    "timeout":        60.0,                                      # 请求超时时间（秒）
    "method":         "POST",                                    # HTTP请求方法
    "concurrency":    2,                                         # 并发连接数，控制同时发送的请求数量
    "retries":        5,                                         # 失败请求的重试次数（0表示不重试）
    "retry_delay":    5.0,                                       # 重试延迟时间（秒）
    "expected_status": None,                                     # 期望的HTTP状态码（None表示不验证）
    "log_level":      "INFO",                                    # 日志级别
    "verbose":        False,                                     # 是否启用详细输出模式
    "json":           True,                                      # 是否使用JSON格式发送数据
    "headers":        {"Content-Type": "application/json"},      # HTTP请求头
    "payload":        None,                                      # 请求体数据（None表示需要从其他地方获取）
    "payload_file":   None,                                      # 包含请求体的文件路径
}

class ConfigManager:
    _instance = None
    _config = DEFAULTS.copy()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def update_random_params(cls):
        """更新随机参数"""
        cls._config['rps'] = random.randint(1, 5)
        cls._config['duration'] = random.uniform(5.0, 20.0)
        cls._config['concurrency'] = random.randint(1, 5)
        
        logging.info(f"随机参数设置: rps={cls._config['rps']}, "
                    f"duration={cls._config['duration']:.1f}s, "
                    f"concurrency={cls._config['concurrency']}")
    
    @classmethod
    def get_config(cls):
        """获取当前配置"""
        return cls._config.copy()  # 返回副本避免意外修改
    
    @classmethod
    def reset_to_defaults(cls):
        """重置为默认配置"""
        cls._config = DEFAULTS.copy()

def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if path:
        cfg_path = Path(path)
    else:
        # default to a file next to this script if it exists
        cfg_path = Path(__file__).with_name('request_simulator.config.json')
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Failed to parse config JSON {cfg_path}: {exc}")
    return {}


def _normalize_headers(h: Any) -> Dict[str, str]:
    if not h:
        return {}
    if isinstance(h, dict):
        return {str(k): str(v) for k, v in h.items()}
    if isinstance(h, list):
        # support ["K=V", "A=B"] style
        out: Dict[str, str] = {}
        for item in h:
            if not isinstance(item, str) or "=" not in item:
                continue
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
        return out
    return {}


def load_payloads_from_config(cfg: Dict[str, Any]) -> Tuple[List[Any], bool, bool]:
    as_json = bool(cfg.get("json", False))
    payload_file = cfg.get("payload_file")
    payload = cfg.get("payload", None)

    if payload_file:
        content = Path(str(payload_file)).read_text(encoding="utf-8")
        if as_json:
            data = json.loads(content)
            if isinstance(data, list):
                return data, as_json, False
            return [data], as_json, False
        lines = [line for line in content.splitlines() if line.strip()]
        return (lines or [""]), as_json, False

    if payload is not None:
        if as_json:
            if isinstance(payload, (dict, list)):
                data = payload
            else:
                data = json.loads(str(payload))
            if isinstance(data, list):
                return data, as_json, False
            return [data], as_json, False
        return [payload], as_json, False

    # No payload configured: fall back to generating one per request
    return [None], as_json, True


def build_request_kwargs(payload: Any, as_json: bool) -> Dict[str, Any]:
    if payload is None:
        return {}
    if as_json:
        return {"json": payload}
    if isinstance(payload, (bytes, bytearray)):
        return {"content": payload}
    return {"data": str(payload)}


def percentile(data: List[float], pct: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    if f == c:
        return sorted_data[f]
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


async def run_load_test(url: str, rps: float, duration: float, timeout: float, cfg: Dict[str, Any]) -> None:
    headers = _normalize_headers(cfg.get("headers"))
    method = str(cfg.get("method", "POST")).upper()
    concurrency = int(cfg.get("concurrency", 1))
    retries = int(cfg.get("retries", 0))
    retry_delay = float(cfg.get("retry_delay", 1.0))
    expected_status = cfg.get("expected_status", None)
    payloads, as_json, use_random_payload = load_payloads_from_config(cfg)

    total_requests = int(duration * rps)
    if total_requests <= 0:
        logging.error("Total requests computed as zero; adjust rps/duration")
        return

    interval = 1.0 / rps if rps > 0 else 0.0
    stats = {
        "success": 0,
        "failure": 0,
        "latencies": [],
        "status_codes": defaultdict(int),
        "errors": defaultdict(int),
    }
    stats_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async with httpx.AsyncClient(timeout=timeout) as client:
        start = time.perf_counter()

        async def fire(idx: int) -> None:
            target_time = start + idx * interval
            sleep_for = target_time - time.perf_counter()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

            if use_random_payload:
                payload = generate_random_payload()
            else:
                payload = payloads[idx % len(payloads)]
            request_kwargs = build_request_kwargs(payload, as_json)
            attempt = 0
            last_error: Optional[str] = None

            while True:
                attempt += 1
                try:
                    async with semaphore:
                        req_start = time.perf_counter()
                        response = await client.request(method, url, headers=headers, **request_kwargs)
                    latency = time.perf_counter() - req_start
                except httpx.RequestError as exc:
                    last_error = str(exc)
                    if attempt > retries + 1:
                        async with stats_lock:
                            stats["failure"] += 1
                            stats["errors"][last_error] += 1
                        return
                    await asyncio.sleep(retry_delay)
                    continue

                status = response.status_code
                ok = status < 400
                if expected_status is not None and status != expected_status:
                    ok = False
                    last_error = f"unexpected status {status}"

                if ok:
                    async with stats_lock:
                        stats["success"] += 1
                        stats["latencies"].append(latency)
                        stats["status_codes"][str(status)] += 1
                    if cfg.get("verbose"):
                        logging.info("Request %d -> %d in %.3fs", idx, status, latency)
                        print("Request %d -> %d in %.3fs", idx, status, latency)
                    return

                last_error = last_error or f"http {status}"
                if attempt > retries + 1:
                    async with stats_lock:
                        stats["failure"] += 1
                        stats["errors"][last_error] += 1
                        stats["status_codes"][str(status)] += 1
                    if cfg.get("verbose"):
                        logging.warning("Request %d failed after retries: %s", idx, last_error)
                    return

                await asyncio.sleep(retry_delay)

        tasks = [asyncio.create_task(fire(i)) for i in range(total_requests)]
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start

    successes = stats["success"]
    failures = stats["failure"]
    total = successes + failures
    logging.info(f"Load test finished: {total} requests in {elapsed:.2f}s ({total / elapsed:.2f} req/s)")
    logging.info(f"Success: {successes}  Failure: {failures}")
    print(f"Load test finished: {total} requests in {elapsed:.2f}s ({total / elapsed:.2f} req/s)")
    print(f"Success: {successes}  Failure: {failures}" )

    if stats["latencies"]:
        logging.info(
            "Latency p50=%.3fs  p95=%.3fs  max=%.3fs",
            statistics.median(stats["latencies"]),
            percentile(stats["latencies"], 95),
            max(stats["latencies"]),
        )
        print(
            "Latency p50=%.3fs  p95=%.3fs  max=%.3fs",
            statistics.median(stats["latencies"]),
            percentile(stats["latencies"], 95),
            max(stats["latencies"]),
        )

    if stats["status_codes"]:
        logging.info("Status codes: %s", dict(stats["status_codes"]))
        print("Status codes: %s", dict(stats["status_codes"]))
    if stats["errors"]:
        logging.info("Top errors: %s", dict(stats["errors"]))
        print("Top errors: %s", dict(stats["errors"]))


def main() -> None:
    # 获取当前配置（包含随机参数）
    current_config = ConfigManager.get_config()
    
    # 原来的配置加载逻辑，但合并当前配置
    cfg_path = os.getenv("SIM_CONFIG_FILE")
    file_cfg = _load_config(cfg_path)
    
    # 合并配置：默认配置 < 文件配置 < 当前随机配置
    cfg = {**DEFAULTS, **file_cfg, **current_config}

    # Extract core parameters
    try:
        url = str(cfg.get("url"))
        rps = float(cfg.get("rps"))
        duration = float(cfg.get("duration"))
        timeout = float(cfg.get("timeout"))
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid core parameter types in config: {exc}")

    # Configure logging based on config
    log_level = getattr(logging, str(cfg.get("log_level", "INFO")).upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")
    if cfg.get("verbose"):
        logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        asyncio.run(run_load_test(url, rps, duration, timeout, cfg))
    except KeyboardInterrupt:  # pragma: no cover
        logging.warning("Interrupted by user")


def job():
    """随机执行模拟机制（带重试机制）"""
    max_retries = 3  # 最大重试次数
    retry_delay = 5  # 重试延迟（秒）
    
    for attempt in range(max_retries):
        try:
            time.sleep(60*random.randint(1,10))  # 随机等待1到10分钟
            logging.info("Starting request simulator with random parameters.")
            ConfigManager.update_random_params()
            logging.info("开始执行模拟"+time.strftime("%m-%d %H:%M:%S", time.localtime()))
            main()
            return
        except Exception as e:
            if attempt < max_retries - 1: 
                time.sleep(retry_delay)
                retry_delay *= 2

def run_scheduler():
    """调度器"""
    schedule.every(1).hours.do(job)
    job()
    while True:
        schedule.run_pending()
        time.sleep(60)  # 每分钟检查一次

if __name__ == "__main__":
    # run_scheduler()
    main()

