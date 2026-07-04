# shared_state3.py
import time
import json
import os
import collections
import numpy as np

# ===================== 状态文件 =====================
EEG_STATE_FILE = "eeg_state.json"
USER_FEEDBACK_FILE = "user_feedback_state.json"   # 保留，兼容旧逻辑
FEEDBACK_QUEUE_FILE = "feedback_queue.jsonl"      # ✅ TTA 新增：反馈队列

# ===================== 全量日志 =====================
EEG_LOG_FILE = "eeg_full_log.jsonl"
FEEDBACK_LOG_FILE = "feedback_full_log.jsonl"
LLM_LOG_FILE = "llm_full_log.jsonl"

# ===================== EEG 原始信号缓存 =====================
EEG_RAW_BUFFER_MAXLEN = 300
_eeg_raw_buffer = collections.deque(maxlen=EEG_RAW_BUFFER_MAXLEN)

# =========================================================
# ===================== EEG 脑电接口 =======================
# =========================================================

def update_eeg_result(result: str, win_start: float = None,
                      win_end: float = None, feat=None):
    data = {
        "result": result,
        "timestamp": time.time()
    }
    with open(EEG_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)

    if win_start is not None and win_end is not None and feat is not None:
        log_item = {
            "win_start": win_start,
            "win_end": win_end,
            "feat": feat.tolist(),
            "pred_label": result,
            "write_ts": time.time()
        }
        with open(EEG_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_item, ensure_ascii=False) + "\n")


def get_eeg_result(max_age=10.0):
    if not os.path.exists(EEG_STATE_FILE):
        return None
    try:
        with open(EEG_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        age = time.time() - data.get("timestamp", 0)
        if age > max_age:
            return None
        return data.get("result")
    except Exception:
        return None


def get_eeg_log_recent(n=100):
    if not os.path.exists(EEG_LOG_FILE):
        return []
    with open(EEG_LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()[-n:]
    return [json.loads(l) for l in lines]


# =========================================================
# ================== EEG 原始信号缓存 ======================
# =========================================================

def append_eeg_raw(window_raw: np.ndarray, win_start: float, win_end: float):
    _eeg_raw_buffer.append({
        "win_start": win_start,
        "win_end": win_end,
        "raw": window_raw.copy()
    })


def get_eeg_raw_recent(n=10):
    recent = list(_eeg_raw_buffer)[-n:]
    out = []
    for item in recent:
        out.append({
            "win_start": item["win_start"],
            "win_end": item["win_end"],
            "raw": item["raw"].tolist()
        })
    return out


# =========================================================
# ================== 用户反馈接口（✅ TTA 重构） =============
# =========================================================

def push_user_feedback(feedback: str, time_range: list, source: str = "manual"):
    """
    source: manual(0) > ai_text(1)
    """
    priority = 0 if source == "manual" else 1
    item = {
        "feedback": feedback,
        "time_range": time_range,
        "source": source,
        "priority": priority,
        "timestamp": time.time(),
        "consumed": False
    }
    with open(FEEDBACK_QUEUE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 兼容旧逻辑（UI仍可读取）
    with open(USER_FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(item, f)

    # 全量日志
    log_item = {
        "ts": item["timestamp"],
        "label": feedback,
        "time_range": time_range,
        "source": source
    }
    with open(FEEDBACK_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_item, ensure_ascii=False) + "\n")


def update_user_feedback(feedback, time_range=None, source="manual"):
    """✅ TTA 新增：兼容旧接口"""
    push_user_feedback(feedback, time_range, source)


def pop_best_feedback(feat_time: float, max_age=30.0):
    """
    按优先级 + 时间对齐消费反馈
    """
    if not os.path.exists(FEEDBACK_QUEUE_FILE):
        return None

    now = time.time()
    items = []

    with open(FEEDBACK_QUEUE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("consumed"):
                continue
            if now - obj["timestamp"] > max_age:
                continue
            items.append(obj)

    if not items:
        return None

    # 手动 > AI，同优先级用最新
    items.sort(key=lambda x: (x["priority"], -x["timestamp"]))

    for item in items:
        tr = item.get("time_range")
        if tr and len(tr) == 2:
            if tr[0] <= feat_time <= tr[1]:
                item["consumed"] = True
                _rewrite_feedback_queue(items)
                return item
        else:
            if abs(feat_time - item["timestamp"]) < 5:
                item["consumed"] = True
                _rewrite_feedback_queue(items)
                return item

    return None


def _rewrite_feedback_queue(items):
    """✅ TTA 新增：回写队列"""
    with open(FEEDBACK_QUEUE_FILE, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def get_user_feedback(max_age=30.0):
    """✅ TTA 新增：兼容旧逻辑（仅UI用）"""
    if not os.path.exists(USER_FEEDBACK_FILE):
        return None
    try:
        with open(USER_FEEDBACK_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        age = time.time() - data.get("timestamp", 0)
        if age > max_age:
            return None
        return data
    except Exception:
        return None


def get_feedback_log_recent(n=100):
    if not os.path.exists(FEEDBACK_LOG_FILE):
        return []
    with open(FEEDBACK_LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()[-n:]
    return [json.loads(l) for l in lines]


# =========================================================
# ================== LLM 输出日志 ==========================
# =========================================================

def log_llm_output(user_input, llm_output, regenerated, segment_evals, ts=None):
    ts = ts or time.time()
    item = {
        "ts": ts,
        "user_input": user_input,
        "llm_output": llm_output,
        "regenerated": regenerated,
        "segment_evals": segment_evals
    }
    with open(LLM_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_llm_log_recent(n=50):
    if not os.path.exists(LLM_LOG_FILE):
        return []
    with open(LLM_LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()[-n:]
    return [json.loads(l) for l in lines]