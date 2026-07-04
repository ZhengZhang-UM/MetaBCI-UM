from psychopy import monitors
import numpy as np
from metabci.brainstim.paradigm import Emotion, paradigm
from metabci.brainstim.framework import Experiment
from metabci.utils.sharedmemory import SharedDict

if __name__ == "__main__":
    # =========================
    # 显示器设置
    # =========================
    mon = monitors.Monitor(
        name="primary_monitor",
        width=59.6,  # 显示器宽度(cm)
        distance=60,  # 被试与屏幕距离(cm)
        verbose=False,
    )
    mon.setSizePix([1920, 1080])
    mon.save()

    bg_color_warm = np.array([0, 0, 0])
    win_size = np.array([1920, 1080])

    # =========================
    # 实验窗口
    # =========================
    ex = Experiment(
        monitor=mon,
        bg_color_warm=bg_color_warm,
        screen_id=0,
        win_size=win_size,
        is_fullscr=False,
        record_frames=False,
        disable_gc=False,
        process_priority="normal",
        use_fbo=False,
    )
    win = ex.get_window(allowGUI=False)

    # =========================
    # Emotion 参数：纯文本评分（简化版）
    # =========================
    emotion_params = {
        # 评分范围（0~10）
        "rating_scale_range": (0, 10),

        # 实验流程定义
        "experiment_setup": {
            "positive_text": "text_rating",
            "negative_text": "text_rating",
            "neutral_text": "text_rating",
        },

        # 每个试次的刺激内容
        "experiment_Stimulus": {
            "positive_text": {
                "rating": "VA",  # VA: Valence-Arousal（愉悦度–唤醒度）
                "text": "请阅读以下内容：\n\n今天天气很好，你顺利完成了一项重要工作。",
                "display_duration": 5,  # 显示5秒
                "pre_stimulus_rest": 1.0,  # 刺激前休息1秒
                "post_stimulus_rest": 0.5,  # 刺激后休息0.5秒
            },
            "negative_text": {
                "rating": "VA",
                "text": "请阅读以下内容：\n\n你在会议中被当众批评，感到有些沮丧。",
                "display_duration": 5,
            },
            "neutral_text": {
                "rating": "VA",
                "text": "请阅读以下内容：\n\n你刚刚收到了期待已久的好消息。",
                "display_duration": 5,
            },
        }
    }

    # =========================
    # 初始化 Emotion 范式
    # =========================
    emotion_obj = Emotion(
        win=win,
        trigger_interval=5,
        **emotion_params
    )

    # =========================
    # LSL / Trigger / 共享内存
    # =========================
    lsl_source_id = "emotion_text_experiment"
    online = True
    shared_dict = SharedDict()

    ex.register_paradigm(
        'Text-based Emotion',
        paradigm,
        VSObject=emotion_obj,
        bg_color=[0, 0, 0],
        pdim='emotion',  # 关键：指定为emotion范式
        port_addr=2,  # port_addr==2 表示虚拟trigger
        nrep=1,
        lsl_source_id=lsl_source_id,
        online=online,
        device_type="Virtual_trigger",
        _buffer=shared_dict
    )

    # =========================
    # 运行实验
    # =========================
    ex.run()