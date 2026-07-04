"""
SEED-IV Dataset Loader for MetaBCI / Brainda
Corrected version: clear separation between subject & session

Data structure:
    eeg_raw_data/
        1/  -> session_0
            1_20131027.mat
            2_20131027.mat
            ...
        2/  -> session_1
        3/  -> session_2

"""

import os
import numpy as np
import scipy.io
from collections import Counter
from pathlib import Path
from mne import create_info
from mne.io import RawArray
from mne.channels import make_standard_montage
from base import BaseDataset
from metabci.brainda.utils.download import mne_data_path


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def time_str_to_seconds(time_str):
    if not isinstance(time_str, str):
        return 0
    parts = time_str.split(":")
    try:
        if len(parts) == 3:
            h, m, s = map(int, parts)
        elif len(parts) == 2:
            h, m, s = 0, int(parts[0]), int(parts[1])
        else:
            return 0
        return h * 3600 + m * 60 + s
    except Exception:
        return 0


# ------------------------------------------------------------
# SEED-IV Dataset
# ------------------------------------------------------------
class SEED_IV(BaseDataset):
    def __init__(
        self,
        path: str = r"E:\archive\seed_iv",
        win_duration: int = 4,
        sessions: list = [0, 1, 2],
    ):
        self.sessions = sessions
        self.session_folder_map = {0: "1", 1: "2", 2: "3"}

        self.raw_data_root = os.path.join(path, "eeg_raw_data")
        if not os.path.exists(self.raw_data_root):
            raise FileNotFoundError(f"Raw data not found: {self.raw_data_root}")

        # ---------- 检查 session 文件夹 ----------
        for s_idx in sessions:
            sess_dir = os.path.join(
                self.raw_data_root, self.session_folder_map[s_idx]
            )
            if not os.path.exists(sess_dir):
                raise FileNotFoundError(f"Session {s_idx} not found: {sess_dir}")

        # ---------- 构建 mat 文件路径 ----------
        # _data_paths[session][subject] = file_url
        self._data_paths = [[None] * 16 for _ in range(3)]  # index 1–15

        for s_idx in range(3):
            sess_dir = os.path.join(
                self.raw_data_root, self.session_folder_map[s_idx]
            )
            for f in sorted(os.listdir(sess_dir)):
                if not f.lower().endswith(".mat"):
                    continue
                # filename: 1_20131027.mat
                subj_str = f.split("_")[0]
                if not subj_str.isdigit():
                    continue
                subj = int(subj_str)
                abs_path = os.path.join(sess_dir, f)
                if os.name == "nt":
                    url = "file://" + abs_path.replace("\\", "/")
                else:
                    url = "file://" + abs_path
                self._data_paths[s_idx][subj] = url

        # ---------- 基本信息 ----------
        self.experiment_name = "SEED-IV"
        self.paradigm = "emotion"
        self.srate = 200

        # ---------- Events ----------
        self.events_name = ["neutral", "sad", "happy", "fear"]
        self._EVENTS = {
            name: (i + 1, (0, win_duration))
            for i, name in enumerate(self.events_name)
        }

        # ---------- Subjects ----------
        self.subjects = list(range(1, 16))

        # ---------- Trial 结构 ----------
        # 24 trials per session
        self._SEQUENCE = np.array([
            0, 1, 2, 3, 0, 1, 2, 3,
            0, 1, 2, 3, 0, 1, 2, 3,
            0, 1, 2, 3, 0, 1, 2, 3
        ])

        self.win_duration = win_duration
        self.win_len = win_duration * self.srate

        # ---------- Channels ----------
        self._CHANNELS = [
            "FP1","FPZ","FP2","AF3","AF4","F7","F5","F3","F1","FZ",
            "F2","F4","F6","F8","FT7","FC5","FC3","FC1","FCZ","FC2",
            "FC4","FC6","FT8","T7","C5","C3","C1","CZ","C2","C4",
            "C6","T8","TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6",
            "TP8","P7","P5","P3","P1","PZ","P2","P4","P6","P8",
            "PO7","PO5","PO3","POZ","PO4","PO6","PO8","POO7","O1","OZ",
            "O2","POO8"
        ]

        super().__init__(
            dataset_code=self.experiment_name,
            subjects=self.subjects,
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=self.srate,
            paradigm=self.paradigm,
        )

    # ----------------------------------------------------------------
    # Data path
    # ----------------------------------------------------------------
    def data_path(
        self,
        subject,
        path=None,
        force_update=False,
        update_path=None,
        proxies=None,
        verbose=None,
    ):
        if subject not in self.subjects:
            raise ValueError(f"Invalid subject id: {subject}")

        if path is None:
            path = os.path.expanduser("~/AssistBCI/mne_Raw_da")
        os.makedirs(path, exist_ok=True)

        # 返回: [session_0_path, session_1_path, session_2_path]
        out = []
        for s_idx in self.sessions:
            url = self._data_paths[s_idx][subject]
            if url is None:
                raise FileNotFoundError(
                    f"Subject {subject} session {s_idx} not found"
                )
            local_path = mne_data_path(
                url,
                self.experiment_name,
                path=path,
                proxies=proxies,
                force_update=force_update,
                update_path=False,
            )
            out.append(local_path)
        return out

    # ----------------------------------------------------------------
    # Load single subject
    # ----------------------------------------------------------------
    def _get_single_subject_data(self, subject, verbose=None):
        montage = make_standard_montage("standard_1005")
        montage.rename_channels({ch: ch.upper() for ch in montage.ch_names})

        ch_names = [ch.upper() for ch in self._CHANNELS] + ["STI 014"]
        ch_types = ["eeg"] * len(self._CHANNELS) + ["stim"]
        info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=self.srate)

        sessions_data = {}

        for s_idx in self.sessions:
            mat_path = self.data_path(subject)[self.sessions.index(s_idx)]

            try:
                mat = scipy.io.loadmat(mat_path)
            except Exception as e:
                if verbose:
                    print(f"[WARN] Failed to load {mat_path}: {e}")
                continue

            runs = {}
            # 每个 session 固定 24 个 trial
            for block_idx in range(24):
                key = self._search(mat, f"_eeg{block_idx + 1}")
                data = mat[key]  # shape: (62, T)
                data = data[:, -6000:]  # 最后 30s

                # 拼接 stim channel
                data = np.vstack([data, np.zeros(data.shape[1])])

                # 打标签
                label = self._SEQUENCE[block_idx] + 1
                trig = [
                    self.win_len * i
                    for i in range(data.shape[1] // self.win_len)
                ]
                data[-1, trig] = label

                raw = RawArray(data, info)
                raw.set_montage(montage)
                runs[f"run_{block_idx + 1:02d}"] = raw

            sessions_data[f"session_{s_idx}"] = runs

        return sessions_data

    # ----------------------------------------------------------------
    # Load all subjects
    # ----------------------------------------------------------------
    def load_all_subjects(self, subjects=None, verbose=True):
        if subjects is None:
            subjects = self.subjects

        all_data = {}
        label_counter = Counter()

        for subj in subjects:
            if verbose:
                print(f">>> Loading Subject {subj}")

            sess_dict = self._get_single_subject_data(subj, verbose=verbose)
            if not sess_dict:
                continue

            all_data[subj] = sess_dict

            for runs in sess_dict.values():
                for raw in runs.values():
                    stim = raw.get_data(picks="STI 014")[0]
                    label_counter.update(stim[stim != 0])

            if verbose:
                n_runs = sum(len(r) for r in sess_dict.values())
                print(f"    Loaded {n_runs} runs")

        if verbose:
            print("\n" + "=" * 60)
            print("✅ All subjects loaded")
            print(f"Subjects: {len(all_data)}")
            print("Label distribution:")
            for k, v in sorted(label_counter.items()):
                print(f"  Label {int(k)}: {v} windows")
            print("=" * 60)

        return all_data

    # ----------------------------------------------------------------
    # Helper
    # ----------------------------------------------------------------
    def _search(self, d, key):
        for k in d.keys():
            if key in k and not k.startswith("__"):
                return k
        raise KeyError(f"Key '{key}' not found in mat file")


# ===================== Test Entry =====================
if __name__ == "__main__":
    DATASET_ROOT = r"E:\archive\seed_iv"

    dataset = SEED_IV(
        path=DATASET_ROOT,
        win_duration=4,
        sessions=[0],  # session 1 only
    )

    all_data = dataset.load_all_subjects(subjects=[1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15])

    # Quick check
    subj = 1
    sess = "session_0"
    run = "run_01"
    raw = all_data[subj][sess][run]
    print(f"\nExample: Subject {subj}, {sess}, {run}")
    print(f"Shape: {raw.get_data().shape}")