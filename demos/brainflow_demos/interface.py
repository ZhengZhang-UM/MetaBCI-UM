# interface12.py
import requests
import gradio as gr
import time
import re
from shared_state import (
    get_eeg_result,
    get_user_feedback,
    update_user_feedback,
    log_llm_output
)

# LLM全局配置，仅使用同一个模型
API_KEY = "dummy"
API_BASE = "http://127.0.0.1:11434/v1"
MODEL = "qwen3:4b"  # 对话、自动评价共用同一个模型
SEGMENT_DELAY = 5
CHARS_PER_SEGMENT = 100
EEG_ANALYSIS_DELAY = 1  # EEG分析显示时间

# 全局变量：记录用户开始输入的时间戳 + 是否已开启计时标记
start_input_time = 0.0
has_input_started = False


def get_current_eeg_evaluation():
    """优先级：手动提交反馈 > AI时段自动评价 > 脑电EEG信号"""
    try:
        fb_info = get_user_feedback(max_age=10.0)
        if fb_info is not None:
            return fb_info["feedback"]
        result = get_eeg_result(max_age=10.0)
        return result if result else "未检测到脑电信号"
    except Exception as e:
        print(f"评价读取异常：{e}")
        return "信号异常"


def submit_user_feedback(select_eval):
    """手动下拉评价提交，固定向前2s作为有效脑电区间"""
    if select_eval is None or select_eval.strip() == "":
        return None, "⚠️ 请先选择「满意/一般/不满意」再提交", 0, 0, 0
    now_ts = time.time()
    time_range = [now_ts - 2.0, now_ts]
    update_user_feedback(select_eval.strip(), time_range=time_range, source="manual")
    return None, f"✅ 手动评价「{select_eval}」提交成功，优先级最高", 0, 0, 0


def auto_start_record(text):
    """输入框文字变化：仅第一次输入文字启动计时，只输出一次日志，后续打字不刷新日志"""
    global start_input_time, has_input_started
    if text.strip() and not has_input_started:
        start_input_time = time.time()
        has_input_started = True
        return "⏱️ 已自动开始记录输入时段"
    return ""


def generate_text_feedback(text: str):
    """共用同一个大模型，对用户输入文本做情绪分类，精简返回日志"""
    global start_input_time, has_input_started
    if not text.strip():
        return "❌ 输入内容为空，无法生成评价"

    # 前置负面关键词硬匹配
    negative_words = [
        "不满意", "不对", "错了", "重新", "重写",
        "太差", "没用", "不好", "看不懂", "答非所问"
    ]
    has_neg = any(w in text for w in negative_words)
    if has_neg:
        label = "不满意"
        end_ts = time.time()
        time_range = [start_input_time, end_ts]
        update_user_feedback(label, time_range=time_range, source="ai_text")
        duration = round(end_ts - start_input_time, 2)
        return f"本次输入时长：{duration}s，🤖 时段AI评价完成，标签：{label}"

    # 优化Prompt，强制仅输出三字标签
    prompt = f"""
严格遵守以下规则，只输出三个汉字，禁止任何多余文字、标点、解释：
规则1：包含满意、认可、不错、很棒、好用、感谢 → 输出：满意
规则2：包含不满、吐槽、错误、重写、看不懂 → 输出：不满意
规则3：普通提问、无情绪表达 → 输出：一般

用户输入内容：{text}
仅输出标签，不要添加任何内容：
"""
    messages = [{"role": "user", "content": prompt}]
    try:
        resp = requests.post(
            f"{API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 1000,
                "stop": ["\n"]
            },
            timeout=30
        )
        resp.raise_for_status()
        raw_out = resp.json()["choices"][0]["message"]["content"].strip()
        match_res = re.findall(r"(满意|一般|不满意)", raw_out)
        label = match_res[0] if match_res else "一般"

        end_ts = time.time()
        time_range = [start_input_time, end_ts]
        update_user_feedback(label, time_range=time_range, source="ai_text")
        duration = round(end_ts - start_input_time, 2)
        return f"本次输入时长：{duration}s，🤖 时段AI评价完成，标签：{label}"
    except Exception as e:
        print(f"自动评价调用异常：{e}")
        return f"❌ 模型调用失败：{str(e)}"


def build_llm_messages(chat_history):
    messages = []
    for usr, bot in chat_history:
        clean = re.sub(r'<span.*?</span>', '', bot)
        clean = re.split(r'<hr style=.*?>', clean)[0]
        messages.append({"role": "user", "content": usr})
        messages.append({"role": "assistant", "content": clean})
    return messages


def call_llm(messages):
    resp = requests.post(
        f"{API_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2000
        },
        timeout=60
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def llm_chat_stream(user_input, chat_history):
    global start_input_time, has_input_started
    feedback_log = ""
    sat_count = 0
    norm_count = 0
    unsat_count = 0

    # 发送时执行AI评价并生成精简日志
    if start_input_time > 0 and has_input_started:
        feedback_log = generate_text_feedback(user_input)
        start_input_time = 0.0
        has_input_started = False

    messages = build_llm_messages(chat_history)
    messages.append({"role": "user", "content": user_input})

    full_display_text = ""
    first_answer = ""
    second_answer = ""
    satisfactory_segs = []
    normal_segs = []
    unsatisfactory_segs = []

    try:
        first_answer = call_llm(messages)
        chat_history.append((user_input, ""))
        text_len = len(first_answer)
        start_idx = 0

        while start_idx < text_len:
            end_idx = start_idx + CHARS_PER_SEGMENT
            seg_text = first_answer[start_idx:end_idx]
            start_idx = end_idx

            # 第一步：显示当前分段
            current_text = full_display_text + seg_text
            chat_history[-1] = (user_input, current_text)
            yield feedback_log, chat_history, sat_count, norm_count, unsat_count
            time.sleep(0.1)  # 短暂延迟确保显示

            # 第二步：显示EEG分析中
            analysis_text = current_text + f"<br><span style='color:#FF4500; font-weight:bold; font-size:0.9em;'>🧠 EEG</span><span style='color:#666; font-size:0.9em;'>信号分析中...</span>"
            chat_history[-1] = (user_input, analysis_text)
            yield feedback_log, chat_history, sat_count, norm_count, unsat_count
            time.sleep(EEG_ANALYSIS_DELAY)

            # 第三步：获取EEG评价并染色
            eval_text = get_current_eeg_evaluation()

            # 根据评价结果染色
            if eval_text == "满意":
                colored_seg = f"<span style='background-color:rgba(16, 185, 129, 0.2); padding:2px 4px; border-radius:3px; border-left:3px solid #10B981;'>{seg_text}</span>"
                satisfactory_segs.append(seg_text)
                sat_count += 1
            elif eval_text == "不满意":
                colored_seg = f"<span style='background-color:rgba(239, 68, 68, 0.2); padding:2px 4px; border-radius:3px; border-left:3px solid #EF4444;'>{seg_text}</span>"
                unsatisfactory_segs.append(seg_text)
                unsat_count += 1
            else:  # 一般
                colored_seg = f"<span style='background-color:rgba(107, 114, 128, 0.1); padding:2px 4px; border-radius:3px; border-left:3px solid #6B7280;'>{seg_text}</span>"
                normal_segs.append(seg_text)
                norm_count += 1

            # 更新完整显示文本，添加染色后的分段和评价结果
            full_display_text += colored_seg
            full_display_text += f"<br><span style='color:#FF4500; font-weight:bold; font-size:0.8em;'>🧠 EEG</span>"
            full_display_text += f"<span style='color:#666; font-size:0.8em;'>评价结果：</span>"
            full_display_text += f"<span style='color:{'#10B981' if eval_text == '满意' else '#EF4444' if eval_text == '不满意' else '#6B7280'}; font-weight:bold; font-size:0.8em;'>{eval_text}</span>"
            full_display_text += "<br>"

            chat_history[-1] = (user_input, full_display_text)
            yield feedback_log, chat_history, sat_count, norm_count, unsat_count
            time.sleep(SEGMENT_DELAY - EEG_ANALYSIS_DELAY - 0.1)

        # 不满意片段重生成
        if unsatisfactory_segs:
            try:
                regeneration_prompt = (
                    "以下是我的原始回答：\n"
                    f"{first_answer}\n\n"
                    "你标记为不满意、需要修改的片段：\n"
                    f"{chr(10).join([f'- {seg}' for seg in unsatisfactory_segs]) if unsatisfactory_segs else '无'}\n\n"
                    "你标记为一般，可小幅优化的片段：\n"
                    f"{chr(10).join([f'- {seg}' for seg in normal_segs]) if normal_segs else '无'}\n\n"
                    "你标记为满意，必须完整保留、不能修改的片段：\n"
                    f"{chr(10).join([f'- {seg}' for seg in satisfactory_segs]) if satisfactory_segs else '无'}\n\n"
                    "要求：完全保留满意片段，仅优化一般片段，大幅修改不满意片段，不改变整体原意，输出完整流畅的新版回答。"
                )
                messages.append({"role": "assistant", "content": first_answer})
                messages.append({"role": "user", "content": regeneration_prompt})
                second_answer = call_llm(messages)

                full_display_text += (
                    "<hr style='margin:10px 0; border-top: 2px dashed #FF4500;'>"
                    "<div style='background: linear-gradient(90deg, rgba(255,69,0,0.1) 0%, rgba(255,69,0,0.05) 100%); padding:10px; border-radius:8px;'>"
                    "<span style='color:#FF4500; font-weight:bold; font-size:0.95em;'>🧠 EEG</span>"
                    "<span style='color:#333; font-weight:bold; font-size:0.95em;'>驱动的重生成回答：</span><br><br>"
                    f"{second_answer}"
                    "</div>"
                )
                chat_history[-1] = (user_input, full_display_text)

                yield feedback_log, chat_history, sat_count, norm_count, unsat_count
            except Exception as reg_err:
                print(f"重生成失败：{reg_err}")
                chat_history[-1] = (user_input, full_display_text)
                yield feedback_log, chat_history, sat_count, norm_count, unsat_count
        else:
            full_display_text += (
                "<hr style='margin:10px 0; border-top: 2px solid #10B981;'>"
                "<div style='background: linear-gradient(90deg, rgba(16,185,129,0.1) 0%, rgba(16,185,129,0.05) 100%); padding:10px; border-radius:8px;'>"
                "<span style='color:#FF4500; font-weight:bold; font-size:0.9em;'>🧠 EEG</span>"
                "<span style='color:#10B981; font-weight:bold; font-size:0.9em;'>综合评价：无不满意片段，回答质量良好</span>"
                "</div>"
            )
            chat_history[-1] = (user_input, full_display_text)
            yield feedback_log, chat_history, sat_count, norm_count, unsat_count

        # ===================== 观测日志：记录 LLM 输出 =====================
        log_llm_output(
            user_input=user_input,
            llm_output=second_answer if unsatisfactory_segs else first_answer,
            regenerated=bool(unsatisfactory_segs),
            segment_evals=[
                {"label": "满意", "count": sat_count},
                {"label": "一般", "count": norm_count},
                {"label": "不满意", "count": unsat_count}
            ]
        )

    except Exception as main_err:
        print(f"生成流程异常：{main_err}")
        err_log = f"❌ 生成流程异常：{str(main_err)}"
        chat_history.append(
            (user_input, full_display_text +
             f"<br><span style='color:red'>生成中途出现异常，以上为已输出内容</span>")
        )
        yield err_log, chat_history, sat_count, norm_count, unsat_count


# 自定义CSS样式 - 重点突出EEG元素
custom_css = """
:root {
    --primary-color: #4f46e5;
    --secondary-color: #6366f1;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --danger-color: #ef4444;
    --eeg-color: #FF4500;  /* 新增EEG主题色 */
    --neutral-color: #6b7280;
    --light-bg: #f8fafc;
    --dark-bg: #1e293b;
    --card-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    --hover-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
}
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 20px !important;
}
.main-title {
    text-align: center;
    margin-bottom: 30px !important;
    color: #1e293b;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.chatbot-container {
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: var(--card-shadow) !important;
    border: 1px solid #e2e8f0 !important;
}
.message.user {
    background-color: var(--primary-color) !important;
    color: white !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 12px 18px !important;
    margin: 8px 0 !important;
    max-width: 85% !important;
}
.message.bot {
    background-color: var(--light-bg) !important;
    color: #1e293b !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 12px 18px !important;
    margin: 8px 0 !important;
    max-width: 85% !important;
    border: 1px solid #e2e8f0 !important;
}
.button-primary {
    background-color: var(--primary-color) !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.button-primary:hover {
    background-color: var(--secondary-color) !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--hover-shadow) !important;
}
.button-secondary {
    background-color: var(--light-bg) !important;
    color: #1e293b !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.button-secondary:hover {
    background-color: #f1f5f9 !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--hover-shadow) !important;
}
.textbox-input {
    border-radius: 12px !important;
    border: 1px solid #e2e8f0 !important;
    padding: 14px 18px !important;
    font-size: 16px !important;
    transition: all 0.2s ease !important;
}
.textbox-input:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1) !important;
    outline: none !important;
}
#feedback_panel {
    background: white !important;
    padding: 20px !important;
    border-radius: 16px !important;
    box-shadow: var(--card-shadow) !important;
    border: 1px solid #e2e8f0 !important;
}
.dropdown-feedback {
    border-radius: 8px !important;
    border: 1px solid #e2e8f0 !important;
    padding: 10px !important;
}
.status-log {
    border-radius: 8px !important;
    border: 1px solid #e2e8f0 !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
}
.stat-box {
    border-radius: 8px !important;
    border: 1px solid #e2e8f0 !important;
    padding: 8px 12px !important;
    font-weight: 500;
}

/* EEG相关样式增强 */
.eeg-highlight {
    color: var(--eeg-color) !important;
    font-weight: bold !important;
    text-shadow: 0 0 2px rgba(255, 69, 0, 0.3);
}
.eeg-badge {
    background: linear-gradient(135deg, var(--eeg-color) 0%, #FF8C00 100%);
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75em;
    font-weight: bold;
    display: inline-block;
    margin-right: 5px;
}
.eeg-analysis-box {
    background-color: rgba(255, 69, 0, 0.05);
    border-left: 3px solid var(--eeg-color);
    padding: 8px 12px;
    margin: 5px 0;
    border-radius: 0 8px 8px 0;
    position: relative;
}
.eeg-analysis-box::before {
    content: "🧠 EEG";
    position: absolute;
    top: -10px;
    left: 10px;
    background: white;
    padding: 0 5px;
    font-size: 0.7em;
    color: var(--eeg-color);
    font-weight: bold;
}
.eeg-pulse {
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.7; }
    100% { opacity: 1; }
}

/* 统计框特殊样式 */
.stat-label {
    font-size: 0.75em;
    color: #666;
    margin-bottom: 2px;
    display: block;
}
.stat-value {
    font-size: 1.5em;
    font-weight: bold;
    text-align: center;
}
.stat-description {
    font-size: 0.7em;
    color: #888;
    text-align: center;
    margin-top: 2px;
}

@media (max-width: 768px) {
    #feedback_panel {
        width: 100% !important;
        margin-top: 20px !important;
    }
    .gradio-container {
        padding: 10px !important;
    }
}
"""

# UI页面构建
with gr.Blocks(
        title="脑电同步自适应对话系统",
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="indigo",
            neutral_hue="slate",
        )
) as demo:
    gr.Markdown(
        "<div style='display:flex; align-items:center; justify-content:center;'>"
        "<span class='eeg-badge eeg-pulse'>🧠 EEG</span>"
        "<span style='font-size:1.8em; font-weight:700;'>脑电同步自适应对话系统</span>"
        "</div>",
        elem_classes="main-title"
    )

    with gr.Row():
        # 左侧聊天区域
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(
                height=600,
                elem_classes="chatbot-container",
                bubble_full_width=False,
                show_label=False,
                avatar_images=(None),
            )
            with gr.Row():
                clear_btn = gr.Button(
                    "🗑️ 清空对话",
                    variant="secondary",
                    elem_classes="button-secondary",
                    size="sm"
                )
            input_box = gr.Textbox(
                label="💬 输入您的问题",
                lines=3,
                placeholder="输入文字将自动开始记录输入时长，回车发送自动生成评价...",
                elem_classes="textbox-input"
            )

        # 右侧反馈面板
        with gr.Column(scale=3, min_width=380, elem_id="feedback_panel"):
            gr.Markdown("### 📊 实时评价面板")

            # 添加EEG标识到面板标题
            gr.Markdown("""
            <div style='display:flex; align-items:center; margin-bottom:10px;'>
                <span class='eeg-badge'>🧠 EEG</span>
                <span style='font-weight:bold;'>神经反馈控制系统</span>
            </div>
            """)

            gr.Markdown("#### 手动评价（最高优先级）")
            feedback_select = gr.Dropdown(
                label="选择评价等级",
                choices=["满意", "一般", "不满意"],
                value=None,
                info="选择后提交，覆盖AI自动评价",
                elem_classes="dropdown-feedback"
            )
            feedback_submit_btn = gr.Button(
                "📝 提交手动评价",
                variant="primary",
                elem_classes="button-primary",
                size="sm"
            )

            gr.Markdown("#### 📈 分段评价统计")

            # 添加统计说明
            gr.Markdown("""
            <div style='background: rgba(255,69,0,0.05); padding:8px; border-radius:6px; margin-bottom:10px;'>
                <span style='color:#FF4500; font-weight:bold; font-size:0.85em;'>🧠 EEG</span>
                <span style='color:#666; font-size:0.85em;'>分段分析结果统计：</span>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("<div class='stat-label'>✅ <span class='eeg-highlight'>EEG</span>满意段落</div>")
                    sat_num = gr.Textbox(
                        value="0",
                        interactive=False,
                        elem_classes="stat-box",
                        container=False,
                        show_label=False
                    )
                    gr.Markdown("<div class='stat-description'>绿色高亮段落数</div>")

                with gr.Column(scale=1):
                    gr.Markdown("<div class='stat-label'>➖ <span class='eeg-highlight'>EEG</span>一般段落</div>")
                    norm_num = gr.Textbox(
                        value="0",
                        interactive=False,
                        elem_classes="stat-box",
                        container=False,
                        show_label=False
                    )
                    gr.Markdown("<div class='stat-description'>灰色边框段落数</div>")

                with gr.Column(scale=1):
                    gr.Markdown("<div class='stat-label'>❌ <span class='eeg-highlight'>EEG</span>不满意段落</div>")
                    unsat_num = gr.Textbox(
                        value="0",
                        interactive=False,
                        elem_classes="stat-box",
                        container=False,
                        show_label=False
                    )
                    gr.Markdown("<div class='stat-description'>红色高亮段落数</div>")

            # 添加统计汇总说明
            gr.Markdown("""
            <div style='margin-top:10px; padding:8px; background:#f8fafc; border-radius:6px; border-left:3px solid #FF4500;'>
                <div style='font-size:0.8em; color:#666;'>
                    <span style='color:#FF4500; font-weight:bold;'>🧠 EEG</span>实时监测脑电信号，
                    将回答按<span style='color:#10B981;'>满意</span>、
                    <span style='color:#6B7280;'>一般</span>、
                    <span style='color:#EF4444;'>不满意</span>分类统计
                </div>
            </div>
            """)

            gr.Markdown("#### 📝 反馈状态日志")
            feedback_status = gr.Textbox(
                label="操作日志",
                interactive=False,
                elem_classes="status-log",
                lines=4,
                max_lines=8
            )

            # 添加EEG状态指示器
            gr.Markdown("""
            <div style='margin-top:15px; padding:10px; background:rgba(255,69,0,0.05); border-radius:8px;'>
                <div style='display:flex; align-items:center;'>
                    <span class='eeg-badge eeg-pulse'>🧠 EEG</span>
                    <span style='font-size:0.9em; color:#666;'>实时脑电信号监测中...</span>
                </div>
                <div style='font-size:0.8em; color:#888; margin-top:5px;'>
                    满意→<span style='color:#10B981;'>绿色高亮</span> | 
                    不满意→<span style='color:#EF4444;'>红色高亮</span> | 
                    一般→<span style='color:#6B7280;'>灰色边框</span>
                </div>
            </div>
            """)

    # 事件绑定
    input_box.change(
        fn=auto_start_record,
        inputs=[input_box],
        outputs=[feedback_status]
    )

    feedback_submit_btn.click(
        fn=submit_user_feedback,
        inputs=[feedback_select],
        outputs=[feedback_select, feedback_status, sat_num, norm_num, unsat_num]
    )

    input_box.submit(
        fn=llm_chat_stream,
        inputs=[input_box, chatbot],
        outputs=[feedback_status, chatbot, sat_num, norm_num, unsat_num]
    )


    def clear_all():
        return [], "", "0", "0", "0"


    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[chatbot, feedback_status, sat_num, norm_num, unsat_num],
        queue=False
    )

demo.queue()
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        inbrowser=True,
        show_error=True,
        show_api=False
    )