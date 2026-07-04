from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from shared_state3 import (
    get_eeg_log_recent,
    get_llm_log_recent,
    get_eeg_raw_recent,
    get_user_feedback,
    get_eeg_result,
    get_feedback_log_recent
)
import time
import uvicorn

app = FastAPI(
    title="EEG–LLM 观测后台",
    description="实时观测脑电解码结果、原始EEG信号、LLM输出与用户反馈",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# ===================== 根路由（修复点） ===================
# =========================================================

@app.get("/", summary="服务首页")
def root():
    return {
        "service": "EEG–LLM Observer API",
        "status": "running",
        "docs": "/docs",
        "dashboard": "/dashboard",
        "note": "推荐使用 /docs 进行可视化观测"
    }


# =========================================================
# ===================== 基础心跳 ==========================
# =========================================================

@app.get("/ping", summary="服务心跳")
def ping():
    return {
        "ok": True,
        "timestamp": time.time()
    }


# =========================================================
# ===================== EEG 接口 ==========================
# =========================================================

@app.get("/eeg/latest", summary="当前EEG判别结果")
def eeg_latest():
    return {
        "label": get_eeg_result(),
        "timestamp": time.time()
    }


@app.get("/eeg/log", summary="EEG判别历史（含时间戳）")
def eeg_log(n: int = 100):
    return get_eeg_log_recent(n)


@app.get("/eeg/raw", summary="EEG原始信号（200Hz，8通道，2s）")
def eeg_raw(n: int = 10):
    """
    返回最近 n 个EEG窗口
    每个窗口: shape (400, 8)
    """
    return get_eeg_raw_recent(n)


# =========================================================
# ===================== 反馈接口 ==========================
# =========================================================

@app.get("/feedback/latest", summary="最新用户反馈")
def feedback_latest():
    fb = get_user_feedback()
    return fb if fb else {}


@app.get("/feedback/log", summary="用户反馈历史")
def feedback_log(n: int = 50):
    return get_feedback_log_recent(n)


# =========================================================
# ===================== LLM 接口 ==========================
# =========================================================

@app.get("/llm/log", summary="LLM输出日志")
def llm_log(n: int = 50):
    return get_llm_log_recent(n)


# =========================================================
# ===================== 科研专用 ==========================
# =========================================================

@app.get("/dashboard", summary="科研一键快照")
def dashboard():
    return {
        "timestamp": time.time(),
        "eeg_current": get_eeg_result(),
        "feedback": get_user_feedback(),
        "llm_last": get_llm_log_recent(1),
        "eeg_raw_last": get_eeg_raw_recent(1)
    }


@app.get("/align", summary="EEG / 反馈 / LLM 时间对齐")
def align_recent(seconds: float = 30.0):
    now = time.time()
    return {
        "time_window_sec": seconds,
        "eeg": [
            e for e in get_eeg_log_recent(2000)
            if now - e["write_ts"] < seconds
        ],
        "feedback": get_user_feedback(),
        "llm": [
            l for l in get_llm_log_recent(200)
            if now - l["ts"] < seconds
        ]
    }


# =========================================================
# ===================== 启动入口 ==========================
# =========================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════╗
║     EEG–LLM Observer API 启动成功       ║
╟──────────────────────────────────────────────────────╢
║  观测面板（推荐）： http://127.0.0.1:8000/docs       ║
║  首页（已修复）：   http://127.0.0.1:8000            ║
║  科研快照：         http://127.0.0.1:8000/dashboard  ║
╚══════════════════════════════════════════════════════╝
""")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )