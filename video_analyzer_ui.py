#!/usr/bin/env python3
"""
视频分析Web界面 - 使用Gradio
"""
import gradio as gr
import requests
import json
import base64
import mimetypes
from pathlib import Path

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

# 默认提示词（35字版本）
DEFAULT_PROMPT = """请用50字以内描述这个广告视频：

要求：
1. 直接从场景和人物开始（如"老师站在黑板前..."）
2. 描述核心动作和话语
3. 说明面向人群和核心问题
4. 简洁清晰，不要用"这个视频"开头"""


def analyze_video(video_file, model_name, custom_prompt, max_tokens):
    """分析视频"""
    try:
        if video_file is None:
            return "❌ 请先上传视频文件", ""

        # 读取视频文件
        video_path = Path(video_file.name)
        file_size_mb = video_path.stat().st_size / 1024 / 1024

        status_msg = f"📤 正在读取视频文件...\n文件: {video_path.name}\n大小: {file_size_mb:.2f} MB\n"

        with open(video_path, 'rb') as f:
            video_data = f.read()

        # 转为base64
        video_base64 = base64.b64encode(video_data).decode('utf-8')
        mime_type = mimetypes.guess_type(str(video_path))[0] or "video/mp4"

        status_msg += f"\n🤖 使用模型: {model_name}\n正在分析...\n"

        # 构建请求
        request_body = {
            "contents": [{
                "parts": [
                    {"text": custom_prompt},
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
            return f"❌ API请求失败\n状态码: {response.status_code}\n{response.text}", ""

        result = response.json()

        # 提取结果
        text_content = result["candidates"][0]["content"]["parts"][0]["text"]
        finish_reason = result["candidates"][0].get("finishReason", "UNKNOWN")

        # 统计信息
        usage = result.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)
        total_tokens = usage.get("totalTokenCount", 0)
        thoughts_tokens = usage.get("thoughtsTokenCount", 0)

        # Token详情
        prompt_details = usage.get("promptTokensDetails", [])
        token_breakdown = "\n".join([
            f"  - {detail.get('modality', 'UNKNOWN')}: {detail.get('tokenCount', 0):,}"
            for detail in prompt_details
        ])

        # 构建统计信息
        stats = f"""📊 Token使用统计
━━━━━━━━━━━━━━━━━━━━
提示词Token: {prompt_tokens:,}
{token_breakdown}

生成Token: {output_tokens:,}
思考Token: {thoughts_tokens:,}
总计Token: {total_tokens:,}

完成原因: {finish_reason}
模型版本: {result.get('modelVersion', 'unknown')}
"""

        # 构建结果
        result_text = f"""✅ 分析完成！

📝 分析结果（{len(text_content)}字）
━━━━━━━━━━━━━━━━━━━━
{text_content}
"""

        return result_text, stats

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f"❌ 发生错误: {str(e)}\n\n详细信息:\n{error_detail}", ""


# 创建Gradio界面
# Gradio 6 移除了 theme 参数，这里使用默认主题以保持兼容
with gr.Blocks(title="视频分析工具 - Gemini Balance") as app:
    gr.Markdown("""
    # 🎬 视频分析工具
    ### 基于 Gemini Balance API
    """)

    with gr.Row():
        with gr.Column(scale=1):
            # 左侧：配置区域
            gr.Markdown("### 📂 选择视频")
            video_input = gr.File(
                label="上传视频文件",
                file_types=[".mp4", ".mov", ".avi", ".mkv"],
                type="filepath"
            )

            gr.Markdown("### 🤖 模型配置")
            model_selector = gr.Dropdown(
                choices=AVAILABLE_MODELS,
                value=AVAILABLE_MODELS[0],
                label="选择Gemini模型",
                info="不同模型的性能和成本不同"
            )

            max_tokens_slider = gr.Slider(
                minimum=512,
                maximum=58192,
                value=12048,
                step=512,
                label="最大输出Token",
                info="控制生成文本的最大长度"
            )

            gr.Markdown("### ✏️ 提示词")
            prompt_input = gr.Textbox(
                label="自定义提示词",
                value=DEFAULT_PROMPT,
                lines=8,
                placeholder="在这里编辑提示词...",
                info="可根据需求修改提示词"
            )

            analyze_btn = gr.Button("🚀 开始分析", variant="primary", size="lg")

        with gr.Column(scale=1):
            # 右侧：结果展示区域
            gr.Markdown("### 📝 分析结果")
            result_output = gr.Textbox(
                label="视频描述",
                lines=10,
                placeholder="分析结果将显示在这里...",
                buttons=["copy"]
            )

            stats_output = gr.Textbox(
                label="Token统计",
                lines=12,
                placeholder="Token使用统计...",
                buttons=["copy"]
            )

    # 示例
    gr.Markdown("""
    ---
    ### 💡 使用提示
    1. **上传视频**: 点击上方文件框上传视频（支持mp4、mov等格式）
    2. **选择模型**: 推荐使用 `gemini-2.5-flash`（快速且经济）
    3. **调整提示词**: 可根据需要修改提示词，控制输出风格和长度
    4. **查看结果**: 点击"开始分析"后，右侧将显示分析结果和Token统计

    ### 📌 注意事项
    - 视频文件建议不超过10MB，以获得更快的响应速度
    - 35字描述适合快速浏览，如需更详细描述可修改提示词
    - Token使用量会影响API成本，请根据需要调整
    """)

    # 绑定事件
    analyze_btn.click(
        fn=analyze_video,
        inputs=[video_input, model_selector, prompt_input, max_tokens_slider],
        outputs=[result_output, stats_output]
    )

# 启动应用
if __name__ == "__main__":
    print("=" * 70)
    print("🎬 视频分析工具启动中...")
    print("=" * 70)
    print(f"\n📡 API地址: {API_BASE}")
    print(f"🔑 API密钥: {API_KEY}")
    print(f"\n🌐 Web界面将在浏览器中自动打开...")
    print("\n提示: 按 Ctrl+C 退出\n")

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
