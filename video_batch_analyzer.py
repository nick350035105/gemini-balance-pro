#!/usr/bin/env python3
"""
批量视频分析和重命名工具 - 专业版
"""
import gradio as gr
import requests
import base64
import mimetypes
import os
import re
from pathlib import Path
from typing import List, Tuple
import pandas as pd

# 配置
API_BASE = "http://localhost:8000"
API_KEY = "sk-demo-token"

# 可用的Gemini模型列表
AVAILABLE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash-exp",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-pro-preview-03-25",
]

# 默认提示词
DEFAULT_PROMPT = """请用50字以内描述这个广告视频：

要求：
1. 直接从场景和人物开始（如"老师站在黑板前..."）
2. 描述核心动作和话语
3. 说明面向人群和核心问题
4. 简洁清晰，不要用"这个视频"开头"""

# 全局状态存储
video_analysis_results = {}


def sanitize_filename(text: str) -> str:
    """清理文本，使其适合作为文件名"""
    # 移除不适合文件名的字符
    text = re.sub(r'[<>:"/\\|?*]', '', text)
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    # 限制长度（防止文件名过长）
    if len(text) > 100:
        text = text[:100]
    return text


def analyze_single_video(video_path: str, model_name: str, prompt: str, max_tokens: int) -> Tuple[bool, str, dict]:
    """分析单个视频"""
    try:
        # 读取视频
        path = Path(video_path)
        with open(path, 'rb') as f:
            video_data = f.read()

        # 转为base64
        video_base64 = base64.b64encode(video_data).decode('utf-8')
        mime_type = mimetypes.guess_type(str(path))[0] or "video/mp4"

        # 构建请求
        request_body = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": video_base64
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": max_tokens,
            }
        }

        # 调用API
        response = requests.post(
            f"{API_BASE}/gemini/v1beta/models/{model_name}:generateContent",
            headers={
                "x-goog-api-key": API_KEY,
                "Content-Type": "application/json"
            },
            json=request_body,
            timeout=180
        )

        if response.status_code != 200:
            return False, f"API错误: {response.status_code}", {}

        result = response.json()
        text_content = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        usage = result.get("usageMetadata", {})

        stats = {
            "prompt_tokens": usage.get("promptTokenCount", 0),
            "output_tokens": usage.get("candidatesTokenCount", 0),
            "total_tokens": usage.get("totalTokenCount", 0),
            "finish_reason": result["candidates"][0].get("finishReason", "UNKNOWN")
        }

        return True, text_content, stats

    except Exception as e:
        return False, f"错误: {str(e)}", {}


def batch_analyze_videos(video_files, model_name, prompt, max_tokens, progress=gr.Progress()):
    """批量分析视频"""
    global video_analysis_results
    video_analysis_results.clear()

    if not video_files:
        return "❌ 请先上传视频文件", None

    results = []
    total = len(video_files)

    for idx, video_file in enumerate(video_files):
        video_path = video_file.name
        video_name = Path(video_path).name

        progress((idx, total), desc=f"分析中: {video_name}")

        success, description, stats = analyze_single_video(
            video_path, model_name, prompt, max_tokens
        )

        if success:
            # 存储结果
            video_analysis_results[video_path] = {
                "original_name": video_name,
                "description": description,
                "stats": stats,
                "path": video_path
            }

            results.append({
                "序号": idx + 1,
                "原文件名": video_name,
                "视频描述": description,
                "Token": stats["total_tokens"],
                "状态": "✅ 成功"
            })
        else:
            results.append({
                "序号": idx + 1,
                "原文件名": video_name,
                "视频描述": description,
                "Token": 0,
                "状态": "❌ 失败"
            })

    df = pd.DataFrame(results)

    summary = f"""
📊 批量分析完成！
━━━━━━━━━━━━━━━━━━━━
总数: {total} 个视频
成功: {len([r for r in results if r['状态'] == '✅ 成功'])} 个
失败: {len([r for r in results if r['状态'] == '❌ 失败'])} 个
总Token: {sum([r['Token'] for r in results]):,}
"""

    return summary, df


def rename_single_video(row_index):
    """重命名单个视频"""
    global video_analysis_results

    if not video_analysis_results:
        return "❌ 请先分析视频", None

    # 获取对应的视频信息
    video_paths = list(video_analysis_results.keys())
    if row_index < 0 or row_index >= len(video_paths):
        return "❌ 无效的行索引", None

    video_path = video_paths[row_index]
    info = video_analysis_results[video_path]

    try:
        old_path = Path(info["path"])
        description = sanitize_filename(info["description"])

        # 保留原扩展名
        extension = old_path.suffix
        new_name = f"{description}{extension}"
        new_path = old_path.parent / new_name

        # 如果文件名已存在，添加序号
        counter = 1
        while new_path.exists() and new_path != old_path:
            new_name = f"{description}_{counter}{extension}"
            new_path = old_path.parent / new_name
            counter += 1

        # 重命名
        old_path.rename(new_path)

        # 更新存储的路径
        video_analysis_results[str(new_path)] = video_analysis_results.pop(video_path)
        video_analysis_results[str(new_path)]["path"] = str(new_path)
        video_analysis_results[str(new_path)]["original_name"] = new_name

        msg = f"✅ 重命名成功！\n\n原文件名:\n{old_path.name}\n\n新文件名:\n{new_name}"

        # 更新表格
        df = generate_current_dataframe()
        return msg, df

    except Exception as e:
        return f"❌ 重命名失败: {str(e)}", None


def rename_all_videos():
    """批量重命名所有视频"""
    global video_analysis_results

    if not video_analysis_results:
        return "❌ 请先分析视频", None

    success_count = 0
    fail_count = 0
    messages = []

    for video_path, info in list(video_analysis_results.items()):
        try:
            old_path = Path(info["path"])

            # 检查文件是否存在
            if not old_path.exists():
                fail_count += 1
                messages.append(f"❌ {old_path.name} - 文件不存在")
                continue

            description = sanitize_filename(info["description"])
            extension = old_path.suffix
            new_name = f"{description}{extension}"
            new_path = old_path.parent / new_name

            # 如果文件名已存在且不是同一个文件，添加序号
            counter = 1
            while new_path.exists() and new_path != old_path:
                new_name = f"{description}_{counter}{extension}"
                new_path = old_path.parent / new_name
                counter += 1

            # 重命名
            if new_path != old_path:
                old_path.rename(new_path)

                # 更新存储
                video_analysis_results[str(new_path)] = video_analysis_results.pop(video_path)
                video_analysis_results[str(new_path)]["path"] = str(new_path)
                video_analysis_results[str(new_path)]["original_name"] = new_name

            success_count += 1
            messages.append(f"✅ {new_name}")

        except Exception as e:
            fail_count += 1
            messages.append(f"❌ {info['original_name']} - {str(e)}")

    summary = f"""
🔄 批量重命名完成！
━━━━━━━━━━━━━━━━━━━━
成功: {success_count} 个
失败: {fail_count} 个

详细信息:
{'─' * 40}
""" + "\n".join(messages)

    df = generate_current_dataframe()
    return summary, df


def generate_current_dataframe():
    """生成当前的DataFrame"""
    global video_analysis_results

    results = []
    for idx, (video_path, info) in enumerate(video_analysis_results.items()):
        results.append({
            "序号": idx + 1,
            "当前文件名": info["original_name"],
            "视频描述": info["description"],
            "Token": info["stats"]["total_tokens"],
            "状态": "✅ 已分析"
        })

    return pd.DataFrame(results)


# 创建Gradio界面
with gr.Blocks(title="批量视频分析与重命名工具", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🎬 批量视频分析与重命名工具
    ### 基于 Gemini Balance API - 专业版
    """)

    with gr.Row():
        # 左侧：配置面板
        with gr.Column(scale=1):
            gr.Markdown("### 📂 上传视频")
            video_files = gr.File(
                label="选择视频文件（支持多选）",
                file_count="multiple",
                file_types=[".mp4", ".mov", ".avi", ".mkv"],
                type="filepath"
            )

            gr.Markdown("### ⚙️ 分析配置")
            model_selector = gr.Dropdown(
                choices=AVAILABLE_MODELS,
                value=AVAILABLE_MODELS[0],
                label="Gemini模型",
                info="推荐使用 gemini-2.5-flash"
            )

            max_tokens = gr.Slider(
                minimum=512,
                maximum=8192,
                value=2048,
                step=512,
                label="最大输出Token"
            )

            prompt_input = gr.Textbox(
                label="提示词",
                value=DEFAULT_PROMPT,
                lines=6,
                placeholder="编辑提示词..."
            )

            analyze_btn = gr.Button("🚀 开始批量分析", variant="primary", size="lg")

        # 右侧：结果面板
        with gr.Column(scale=2):
            gr.Markdown("### 📊 分析结果")
            status_output = gr.Textbox(
                label="状态信息",
                lines=6,
                placeholder="分析状态将显示在这里..."
            )

            results_table = gr.Dataframe(
                label="视频分析结果",
                headers=["序号", "当前文件名", "视频描述", "Token", "状态"],
                interactive=False,
                wrap=True
            )

    # 重命名操作区
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🔄 重命名操作")
            gr.Markdown("**单个重命名**: 输入表格中的序号")

            with gr.Row():
                row_index_input = gr.Number(
                    label="视频序号",
                    value=1,
                    minimum=1,
                    precision=0
                )
                rename_single_btn = gr.Button("重命名此视频", variant="secondary")

        with gr.Column(scale=1):
            gr.Markdown("### 🔄 批量重命名")
            gr.Markdown("**批量操作**: 重命名所有已分析的视频")
            rename_all_btn = gr.Button("🔄 重命名全部视频", variant="primary", size="lg")

    # 重命名结果
    rename_output = gr.Textbox(
        label="重命名结果",
        lines=8,
        placeholder="重命名结果将显示在这里...",
        show_copy_button=True
    )

    # 使用说明
    gr.Markdown("""
    ---
    ### 💡 使用说明

    #### 第一步：上传并分析
    1. 点击"选择视频文件"，支持**同时上传多个视频**
    2. 选择Gemini模型（推荐 `gemini-2.5-flash`）
    3. 根据需要调整提示词（默认生成50字描述）
    4. 点击"🚀 开始批量分析"

    #### 第二步：重命名文件
    - **单个重命名**: 在"视频序号"输入框输入序号（如1、2、3），点击"重命名此视频"
    - **批量重命名**: 点击"🔄 重命名全部视频"，一键重命名所有视频

    #### 📌 注意事项
    - 视频文件名将被替换为生成的描述内容
    - 原始扩展名（如.mp4）会保留
    - 如果文件名重复，会自动添加序号（如: 描述_1.mp4）
    - 重命名操作不可撤销，建议先备份重要文件

    #### 🎯 适用场景
    - 批量整理广告素材库
    - 为视频文件添加语义化命名
    - 快速生成视频内容标签
    """)

    # 绑定事件
    analyze_btn.click(
        fn=batch_analyze_videos,
        inputs=[video_files, model_selector, prompt_input, max_tokens],
        outputs=[status_output, results_table]
    )

    rename_single_btn.click(
        fn=lambda idx: rename_single_video(int(idx) - 1),  # 用户输入从1开始，内部从0开始
        inputs=[row_index_input],
        outputs=[rename_output, results_table]
    )

    rename_all_btn.click(
        fn=rename_all_videos,
        outputs=[rename_output, results_table]
    )


if __name__ == "__main__":
    print("=" * 70)
    print("🎬 批量视频分析与重命名工具 - 专业版")
    print("=" * 70)
    print(f"\n📡 API地址: {API_BASE}")
    print(f"🔑 API密钥: {API_KEY}")
    print(f"\n🌐 Web界面启动中...")
    print("\n功能:")
    print("  ✓ 批量上传视频")
    print("  ✓ 自动生成描述")
    print("  ✓ 单个/批量重命名")
    print("\n提示: 按 Ctrl+C 退出\n")

    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        inbrowser=True
    )
