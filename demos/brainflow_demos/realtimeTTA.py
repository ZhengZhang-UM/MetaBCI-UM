# realtimeTTA8.py
from metabci.brainflow.workers import ProcessWorker
import numpy as np
import time
import torch
import torch.nn.functional as F
from scipy import signal
from shared_state4 import (
    update_eeg_result,
    append_eeg_raw,
    pop_best_feedback
)
import collections

# ====================== 全局配置 ======================
SAMPLE_RATE = 1000
WINDOW_SEC = 5
WINDOW_LEN = int(SAMPLE_RATE * WINDOW_SEC)
WINDOW_STEP_SEC = 0.5
WINDOW_STEP = int(SAMPLE_RATE * WINDOW_STEP_SEC)
NUM_CHAN = 8
CLASSES = ["不满意", "一般", "满意"]
LABEL_MAP = {"不满意": 0, "一般": 1, "满意": 2}

MAX_TRAIN_SAMPLES = 200
USER_FEEDBACK_VALID_TIME = 30.0
TTA_UPDATE_INTERVAL = 1.0
TIME_DELAY_OFFSET = 0.15
TARGET_SR = 200

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EEGNet(torch.nn.Module):
    def __init__(self, n_chan=8, n_samples=200, n_class=3):
        super().__init__()
        self.firstconv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, (1, 64), padding=(0, 32)),
            torch.nn.BatchNorm2d(8),
            torch.nn.ELU(),
            torch.nn.AvgPool2d((1, 4)),
            torch.nn.Dropout(0.25)
        )
        self.depthwise = torch.nn.Sequential(
            torch.nn.Conv2d(8, 16, (n_chan, 1)),
            torch.nn.BatchNorm2d(16),
            torch.nn.ELU(),
            torch.nn.AvgPool2d((1, 8)),
            torch.nn.Dropout(0.25)
        )
        self.classify = torch.nn.Linear(16 * (n_samples // 32), n_class)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = x.flatten(1)
        return self.classify(x)

# ====================== Worker ======================
class EEGTTAWorker(ProcessWorker):
    def __init__(self, timeout=5e-2, name="eeg_tta_worker"):
        super().__init__(timeout=timeout, name=name)

        self.model = None
        self.optimizer = None
        self.eeg_data_buffer = []
        self.last_tta_update_time = 0.0
        self.current_win_start = None
        self.current_win_end = None

    # ===== TTA =====
    def pre(self):
        print(f"[{self.worker_name}] 加载 best_model.pth")
        self.model = EEGNet(n_chan=NUM_CHAN, n_samples=TARGET_SR * WINDOW_SEC)
        self.model.load_state_dict(
            torch.load("best_model.pth", map_location=DEVICE)
        )
        self.model.to(DEVICE)
        self.model.eval()

        # ✅ TTA：只优化 BN 参数
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-3
        )

    # ===== TTA =====
    def _tta_step(self, x):
        """熵最小化 TTA（无监督）"""
        self.model.train()
        logits = self.model(x)
        prob = F.softmax(logits, dim=1)
        entropy = -(prob * prob.log()).sum(dim=1).mean()

        self.optimizer.zero_grad()
        entropy.backward()
        self.optimizer.step()
        self.model.eval()

        return prob.detach()

    def consume(self, data):
        now_time = time.time()

        # -------- EEG 缓存 --------
        eeg_raw = np.array(data)[:, :NUM_CHAN]
        self.eeg_data_buffer.extend(eeg_raw)

        if len(self.eeg_data_buffer) < WINDOW_LEN:
            return

        window_data = np.array(self.eeg_data_buffer[:WINDOW_LEN])

        # -------- 时间对齐 --------
        raw_win_end = now_time
        raw_win_start = raw_win_end - WINDOW_SEC
        real_win_start = raw_win_start - TIME_DELAY_OFFSET
        real_win_end = raw_win_end - TIME_DELAY_OFFSET
        self.current_win_start = real_win_start
        self.current_win_end = real_win_end

        # -------- 预处理（与你原来一致） --------
        x = window_data.T[np.newaxis, :NUM_CHAN, ...]
        dec = SAMPLE_RATE // TARGET_SR
        x = signal.decimate(x, dec)
        x = signal.detrend(x)
        b_notch, a_notch = signal.iirnotch(50, 4, TARGET_SR)
        x = signal.filtfilt(b_notch, a_notch, x)
        nyq = TARGET_SR / 2
        b_band, a_band = signal.butter(4, [1 / nyq, 75 / nyq], btype='band')
        x = signal.filtfilt(b_band, a_band, x)

        window_proc = x.squeeze(0).T  # (200, 8)
        append_eeg_raw(window_proc, real_win_start, real_win_end)

        # ===== TTA ===== 转 tensor
        x_tensor = torch.tensor(
            window_proc.T[None, None, ...],  # (1, 1, 8, 200)
            dtype=torch.float32
        ).to(DEVICE)

        # -------- 推理 --------
        with torch.no_grad():
            logits = self.model(x_tensor)
            prob = F.softmax(logits, dim=1)
        pred = prob.argmax(dim=1).item()
        pred_label = CLASSES[pred]

        update_eeg_result(pred_label, real_win_start, real_win_end, prob.cpu().numpy())

        print(f"[{self.worker_name}] 预测：{pred_label} | "
              f"[{real_win_start:.2f}, {real_win_end:.2f}]")

        # ===== TTA：无监督熵最小化 =====
        if now_time - self.last_tta_update_time < TTA_UPDATE_INTERVAL:
            return

        # 1️⃣ 优先使用用户反馈（强监督）
        fb_info = pop_best_feedback(
            (real_win_start + real_win_end) / 2.0,
            max_age=USER_FEEDBACK_VALID_TIME
        )

        if fb_info is not None and fb_info["feedback"] in LABEL_MAP:
            fb_label = LABEL_MAP[fb_info["feedback"]]
            label_tensor = torch.tensor([fb_label], device=DEVICE)

            self.model.train()
            logits = self.model(x_tensor)
            loss = F.cross_entropy(logits, label_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.eval()

            print(f"[{self.worker_name}] ✅ 反馈 TTA（CE Loss={loss.item():.4f}）")

        else:
            # 2️⃣ 无反馈 → 熵最小化（标准 TTA）
            prob = self._tta_step(x_tensor)
            entropy = -(prob * prob.log()).sum(dim=1).mean().item()
            print(f"[{self.worker_name}] ✅ 无监督 TTA（Entropy={entropy:.4f})")

        self.last_tta_update_time = now_time

    def post(self):
        print(f"[{self.worker_name}] 清理资源")
        self.eeg_data_buffer.clear()

# ====================== 主程序（不变） ======================
if __name__ == "__main__":
    from metabci.brainflow.amplifiers import NeuroDance, Marker

    tta_worker = EEGTTAWorker(timeout=5e-2)
    marker = Marker(interval=[0, 2], srate=SAMPLE_RATE)

    d = NeuroDance(
        device_address=('127.0.0.1', 8899),
        srate=1000,
        num_chans=8
    )

    d.connect_tcp()
    d.register_worker('eeg_tta_worker', tta_worker, marker)
    d.up_worker('eeg_tta_worker')

    time.sleep(2)
    input('press any key to start\n')
    d.start_trans()

    input('press any key to close\n')
    d.down_worker('eeg_tta_worker')
    d.stop_trans()
    d.close_connection()
    d.clear()
    print('bye')