#!/usr/bin/env python3
"""
批量视频分析和重命名工具 - 专业版 V2
支持本地目录批量处理 + 分镜脚本解析
"""
import gradio as gr
import requests
import base64
import mimetypes
import os
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import pandas as pd
import shutil

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

# 思考模式支持的模型
THINKING_SUPPORTED_MODELS = {
    "gemini-2.5-flash": "optional",  # 可选
    "gemini-2.5-pro": "required",    # 强制启用
    "gemini-2.5-pro-preview-03-25": "required",
}

# 默认提示词
DEFAULT_PROMPT = """请用50字以内描述这个广告视频：

要求：
1. 直接从场景和人物开始（如"老师站在黑板前..."）
2. 描述核心动作和话语
3. 说明面向人群和核心问题
4. 简洁清晰，不要用"这个视频"开头"""

# 分镜脚本解析提示词 - 算法工程师优化版
STORYBOARD_PROMPT = """你是一个专业的视频分镜分析系统。你的任务是按照时间顺序解析视频，输出结构化的分镜脚本数据。

## 核心要求
你必须返回一个**纯JSON数组**，不要添加任何解释性文字、markdown标记、或代码块标识。

## 输出格式模板
[
  {
    "boardNo": 1,
    "subtitle": "字幕内容",
    "speaker": "说话人",
    "straightOn": true,
    "floatingScreen": false,
    "scene": "画面描述"
  }
]

## 字段规范

### 1. boardNo (整数，必填)
- 从1开始的连续递增编号
- 每个镜头切换时递增1
- 不允许跳号或重复

### 2. subtitle (字符串，必填)
- **提取规则**: 仅提取视频中实际出现的字幕文字
- **空字幕处理**: 如果该镜头无字幕，填入空字符串 ""
- **多行字幕**: 使用 \\\\ 分隔不同行或时间段的字幕
- **禁止行为**: 不要根据配音或画面内容编造字幕
- 示例: "欢迎来到本期节目\\\\今天我们聊聊AI技术"

### 3. speaker (字符串，必填)
- 识别并标注说话人的身份
- 优先级: 字幕标注 > 画面判断 > 声音特征
- 无法识别时填写: "未知说话人"
- 多人同时说话时用"/"分隔: "主持人/嘉宾"
- 示例: "老师"、"博主"、"旁白"

### 4. straightOn (布尔值，必填)
- **判断标准**: 说话人是否以全屏方式正对镜头进行口播
- **true 的条件**:
  * 人物占据画面主体（≥60%）
  * 人脸正对镜头（±30°范围内）
  * 人物在进行口播表达
- **false 的情况**:
  * 人物为浮窗/画中画形式（即使正视镜头）
  * 人物侧面、背面或其他角度
  * 纯画面展示无人物
  * 人物在画面中但未口播
- 使用小写: true / false

### 5. floatingScreen (布尔值，必填)
- **判断标准**: 说话人是否以浮窗形式出现
- **true 的条件**:
  * 人物以小窗口形式叠加在主画面上
  * 画中画效果
  * 分屏展示中的人物窗口
- **false 的情况**:
  * 全屏人物
  * 纯画面展示
  * 人物占据主要画面
- 使用小写: true / false

### 6. scene (字符串，必填)
- **描述结构**: [主体] + [动作/状态] + [环境/道具] + [文字信息]
- **描述粒度**: 30-80字，重点突出关键信息
- **必须包含**:
  * 画面主体（人物/物体/场景）
  * 关键动作或状态
  * 重要的视觉元素（图表、文字、产品等）
- **可选包含**:
  * 背景环境描述
  * 色调或氛围
  * 镜头运动（推拉摇移）
- 示例: "老师站在黑板前，手持教鞭指向板书'方程式'，背景是整洁的教室，黑板左侧有课程表"

## 分镜切换判断标准
识别以下情况为新的分镜:
1. 镜头角度或景别变化（特写→中景→远景）
2. 场景或背景切换
3. 说话人变化
4. 画面主体内容发生显著变化
5. 明显的转场效果（淡入淡出、切换等）

## 边界情况处理

### 情况A: 无字幕镜头
{
  "boardNo": 1,
  "subtitle": "",
  "speaker": "旁白",
  "straightOn": false,
  "floatingScreen": false,
  "scene": "城市夜景航拍，高楼大厦灯光璀璨"
}

### 情况B: 浮窗口播
{
  "boardNo": 2,
  "subtitle": "大家好，我是主播",
  "speaker": "主播",
  "straightOn": false,
  "floatingScreen": true,
  "scene": "左下角浮窗显示主播正面特写，背景是产品展示画面"
}

### 情况C: 纯画面展示
{
  "boardNo": 3,
  "subtitle": "",
  "speaker": "无",
  "straightOn": false,
  "floatingScreen": false,
  "scene": "产品3D建模旋转展示，白色背景，聚光灯效果"
}

## 质量检查清单
输出前请确认:
- [ ] 返回的是纯JSON数组（无任何前后缀文字）
- [ ] boardNo 从1开始连续递增
- [ ] 所有字段都已填写（不允许null或undefined）
- [ ] 布尔值使用小写 true/false
- [ ] 字符串中的特殊字符已正确转义
- [ ] subtitle 仅包含实际字幕，无编造内容
- [ ] straightOn 和 floatingScreen 的判断符合标准

现在开始分析视频并输出JSON数组:"""

# 视频提示词反推提示词
VIDEO_PROMPT_REVERSE_PROMPT = """# 角色与目标
你是一名"AI视频提示词逆向工程专家"，世界级的专家，擅长分析视频内容，并将其转化为适用于AI视频生成模型（如Sora, Runway, Pika等）的高度详细、结构化且高效的提示词。你的首要目标是接收用户提供的视频，并产出一个专业级的提示词，该提示词能够以最高的保真度复现原视频的视觉风格、叙事、情感基调和关键动作。

# 核心工作流
你的工作流程在接收到用户视频后
你将从以下五个关键维度，在内部静默且细致地分析视频。你不会直接输出这份分析，而是将其作为最终提示词的基础。
1.  **概念核心：**
    *   **主题与叙事：** 视频的核心信息、故事或目的是什么（例如：产品广告、情感片段、教育内容）？
    *   **情绪与氛围：** 主要的情感基调是什么（例如：宁静、混乱、史诗、幽默、忧郁）？
2.  **视觉元素：**
    *   **场景与环境：** 故事发生在何时何地？识别具体时间、地点、建筑、景观和关键环境细节。
    *   **主体与人物：** 焦点是谁或什么？描述其外貌、衣着、年龄、物种和情绪表达。
    *   **物体与道具：** 场景中有哪些关键物体？它们如何与主体或场景互动？
3.  **动态与摄影：**
    *   **动作序列：** 按时间顺序分解关键动作和事件。使用强有力的动词。
    *   **镜头语言：** 识别景别（特写、全景）、摄像机角度（低角度、高角度）和运镜方式（固定、手持、跟踪镜头）。
4.  **美学与风格：**
    *   **视觉风格：** 是写实影像、二维动画、三维CG、极简主义，还是超现实主义？
    *   **光影：** 描述光的质量和方向（例如：柔和的晨光、刺眼的直射光、情绪化的霓虹灯）。
    *   **调色板：** 识别主色调、饱和度和对比度（例如：鲜艳饱和、低饱和度柔和）。
5.  **听觉信息：**
    *   **音乐：** 分析背景音乐的类型、节奏、乐器和情感影响。
    *   **音景：** 识别关键音效、对话和环境噪音。
6.  **注意要点：**
    *   **文字处理：**原则上你描述的视频画面中的不包含任何文字、字幕。如果原视频的主体是文字，那么你需要概括性的描述（如一本语文书、小说、书画、指示牌等概括性的替代具体的文字）
    *   **音乐开篇高潮：**在"音频信息"部分，你必须推荐一段**从一开始就进入高潮或高能量点的背景音乐**。如果原视频没有音乐，你必须创造性地构思一段符合视频氛围的、同样是开篇即高潮的配乐。

# 输出格式与规则
你必须严格遵守以下结构和新增的核心规则来生成最终输出，只输出结果，不要做任何解释和描述。
    ```markdown
    ### **[视频分析摘要]**
    *(用一句话简要概括视频的核心概念和风格。)*
    ---
    ### **[详细的AI视频提示词]**

    **英文提示词 (AI模型优先使用):**
   **Overall Concept & Style:** *(描述核心主题、情绪和视觉风格。)*
   **Scene & Environment:** *(详述场景、光线和氛围。)*
   **Subjects & Characters:** *(描述主要对象及其外观/动作。)*
   **Key Action Sequence:** *(分步拆解关键事件。)*
   **Cinematography & Aesthetics:** *(指明摄影手法、角度、色彩和整体视觉感受。)*
   **Audio Profile (For reference/post-production):**
   **Background Music:** *(推荐一段开篇即高潮的音乐，描述其类型、情绪和乐器。)*
   **Soundscape:** *(描述关键音效或对话。)*

    **中文释义 (供用户理解):**
   **整体概念与风格：** *(对应的中文翻译)*
   **场景与环境：** *(对应的中文翻译)*
   **主体与人物：** *(对应的中文翻译)*
   **关键动作序列：** *(对应的中文翻译)*
   **摄影与美学：** *(对应的中文翻译)*
   **音频信息：**
   **背景音乐：** *(对应的中文翻译)*
   **音景：** *(对应的中文翻译)*
    ```
**4. 语言：**
始终优先提供**英文提示词**，因为这是当前多数AI视频模型效果最好的语言。随后必须附上清晰的**中文释义**，方便用户理解。

**5. 描述性：**
使用丰富、生动的语言。在适当的地方使用具体的形容词、副词和专业术语。

**6. 清晰度：**
确保提示词逻辑清晰、无歧义，便于AI模型解析和执行。
"""

# 默认视频目录
DEFAULT_VIDEO_DIR = "/Users/chen/ClaudeCode/工作项目/行业素材库/前贴提取结果"

# 全局状态存储
video_analysis_results = {}
storyboard_results = {}
video_prompt_results = {}


def sanitize_filename(text: str) -> str:
    """清理文本，使其适合作为文件名"""
    # 移除不适合文件名的字符
    text = re.sub(r'[<>:"/\\|?*\n\r\t]', '', text)
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    # 移除首尾的点号
    text = text.strip('.')
    # 限制长度
    if len(text) > 80:
        text = text[:80]
    return text


def update_thinking_mode_state(model_name: str):
    """根据选择的模型更新思考模式复选框的状态"""
    thinking_support = THINKING_SUPPORTED_MODELS.get(model_name, "disabled")

    if thinking_support == "required":
        # Pro模型：强制启用思考模式，复选框勾选且禁用
        return gr.update(value=True, interactive=False, info="✓ Pro模型默认启用思考模式且无法关闭")
    elif thinking_support == "optional":
        # Flash模型：用户可选，复选框可用
        return gr.update(value=False, interactive=True, info="⚠️ 思考模式会消耗1000-2000个额外Token，可能导致内容为空。推荐关闭。")
    else:
        # 其他模型：不支持思考模式，复选框未勾选且禁用
        return gr.update(value=False, interactive=False, info="✗ 此模型不支持思考模式")


def get_video_files(directory: str) -> List[str]:
    """获取目录下的所有视频文件"""
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv'}
    video_files = []

    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        return []

    for file_path in dir_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            video_files.append(str(file_path))

    return sorted(video_files)


def parse_storyboard_json(text: str) -> List[Dict[str, Any]]:
    """解析分镜脚本JSON，支持多种格式"""
    try:
        # 移除可能的markdown代码块标记
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # 尝试解析JSON
        data = json.loads(text)

        # 确保返回列表
        if isinstance(data, dict):
            if "storyboard" in data:
                data = data["storyboard"]
            elif "scenes" in data:
                data = data["scenes"]
            else:
                data = [data]

        return data if isinstance(data, list) else []
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print(f"原始文本: {text[:500]}")
        return []


def analyze_storyboard(video_path: str, model_name: str, max_tokens: int, enable_thinking: bool = False, prompt: str = None) -> Tuple[bool, List[Dict], dict]:
    """分析视频的分镜脚本"""
    try:
        # 使用传入的提示词或默认提示词
        if not prompt:
            prompt = STORYBOARD_PROMPT

        # 读取视频
        path = Path(video_path)
        file_size = path.stat().st_size

        # 文件大小限制 (50MB)
        if file_size > 50 * 1024 * 1024:
            return False, [], {"error": "文件过大（>50MB）"}

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
                "temperature": 0.3,  # 降低温度以获得更稳定的JSON输出
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": max_tokens,
            }
        }

        # 根据思考模式设置
        if not enable_thinking:
            request_body["systemInstruction"] = {
                "parts": [{"text": "请直接给出JSON格式的分镜脚本，不要展示思考过程。"}]
            }

        # 调用API
        response = requests.post(
            f"{API_BASE}/gemini/v1beta/models/{model_name}:generateContent",
            headers={
                "x-goog-api-key": API_KEY,
                "Content-Type": "application/json"
            },
            json=request_body,
            timeout=300  # 分镜脚本解析可能需要更长时间
        )

        if response.status_code != 200:
            return False, [], {"error": f"API错误: {response.status_code}"}

        result = response.json()

        # 检查是否有内容
        if not result.get("candidates") or not result["candidates"][0].get("content"):
            return False, [], {"error": "API返回空内容"}

        text_content = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        usage = result.get("usageMetadata", {})

        # 解析JSON
        storyboard = parse_storyboard_json(text_content)

        if not storyboard:
            return False, [], {"error": "JSON解析失败", "raw_text": text_content[:500]}

        stats = {
            "prompt_tokens": usage.get("promptTokenCount", 0),
            "output_tokens": usage.get("candidatesTokenCount", 0),
            "total_tokens": usage.get("totalTokenCount", 0),
            "finish_reason": result["candidates"][0].get("finishReason", "UNKNOWN"),
            "scene_count": len(storyboard)
        }

        return True, storyboard, stats

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return False, [], {"error": f"错误: {str(e)}", "detail": error_detail}


def analyze_single_video(video_path: str, model_name: str, prompt: str, max_tokens: int, enable_thinking: bool = False) -> Tuple[bool, str, dict]:
    """分析单个视频"""
    try:
        # 读取视频
        path = Path(video_path)
        file_size = path.stat().st_size

        # 文件大小限制 (50MB)
        if file_size > 50 * 1024 * 1024:
            return False, "文件过大（>50MB）", {}

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

        # 根据思考模式设置，添加systemInstruction来控制
        if not enable_thinking:
            # 禁用思考模式：明确指示模型不要进行思考过程
            request_body["systemInstruction"] = {
                "parts": [{"text": "请直接给出答案，不要展示思考过程。"}]
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

        # 检查是否有内容
        if not result.get("candidates") or not result["candidates"][0].get("content"):
            return False, "API返回空内容", {}

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


def scan_directory(directory_path):
    """扫描目录，获取视频列表"""
    if not directory_path or not os.path.exists(directory_path):
        return f"❌ 目录不存在: {directory_path}", None

    video_files = get_video_files(directory_path)

    if not video_files:
        return f"❌ 目录中没有找到视频文件: {directory_path}", None

    df_data = []
    for idx, video_path in enumerate(video_files, 1):
        file_size = Path(video_path).stat().st_size / 1024 / 1024
        df_data.append({
            "序号": idx,
            "文件名": Path(video_path).name,
            "大小(MB)": f"{file_size:.2f}",
            "状态": "待分析"
        })

    df = pd.DataFrame(df_data)
    message = f"✅ 扫描完成！\n找到 {len(video_files)} 个视频文件"

    return message, df


def batch_analyze_videos(directory_path, model_name, prompt, max_tokens, enable_thinking):
    """批量分析目录中的视频 - 流式展示结果"""
    global video_analysis_results
    video_analysis_results.clear()

    if not directory_path or not os.path.exists(directory_path):
        yield "❌ 请输入有效的目录路径", None
        return

    video_files = get_video_files(directory_path)

    if not video_files:
        yield "❌ 目录中没有视频文件", None
        return

    results = []
    total = len(video_files)

    # 初始状态
    initial_summary = f"""
🚀 开始批量视频描述分析
━━━━━━━━━━━━━━━━━━━━
📂 目录: {directory_path}
📊 总数: {total} 个视频
⏳ 准备开始处理...
"""
    yield initial_summary, None

    for idx, video_path in enumerate(video_files):
        video_name = Path(video_path).name

        # 检查文件是否存在
        if not Path(video_path).exists():
            # 文件不存在，跳过
            current_num = idx + 1
            results.append({
                "序号": current_num,
                "文件名": video_name,
                "大小(MB)": "N/A",
                "视频描述": "文件不存在",
                "Token": 0,
                "状态": "❌ 失败"
            })
            continue

        file_size = Path(video_path).stat().st_size / 1024 / 1024

        # 处理前显示状态
        current_num = idx + 1
        progress_percent = int(current_num / total * 100)
        progress_bar = "█" * (progress_percent // 5) + "░" * (20 - progress_percent // 5)

        # 添加loading动画效果
        loading_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        loading_icon = loading_frames[idx % len(loading_frames)]

        processing_summary = f"""
{loading_icon} 正在分析视频 {current_num}/{total}
━━━━━━━━━━━━━━━━━━━━
📂 目录: {directory_path}
📊 进度: [{progress_bar}] {current_num}/{total} ({progress_percent}%)
📹 当前: {video_name} ({file_size:.1f}MB)
⏳ 分析中，请稍候...
"""
        yield processing_summary, pd.DataFrame(results) if results else None

        success, description, stats = analyze_single_video(
            video_path, model_name, prompt, max_tokens, enable_thinking
        )

        if success:
            video_analysis_results[video_path] = {
                "original_name": video_name,
                "original_path": video_path,
                "description": description,
                "stats": stats,
            }

            results.append({
                "序号": current_num,
                "文件名": video_name,
                "大小(MB)": f"{file_size:.2f}",
                "视频描述": description,
                "Token": stats["total_tokens"],
                "状态": "✅ 成功"
            })
        else:
            results.append({
                "序号": current_num,
                "文件名": video_name,
                "大小(MB)": f"{file_size:.2f}",
                "视频描述": description,
                "Token": 0,
                "状态": "❌ 失败"
            })

        # 分析完成后立即更新结果
        df = pd.DataFrame(results)
        success_count = len([r for r in results if r['状态'] == '✅ 成功'])
        fail_count = len([r for r in results if r['状态'] == '❌ 失败'])
        total_tokens = sum([r['Token'] for r in results])

        result_summary = f"""
✅ 已完成 {current_num}/{total} 个视频
━━━━━━━━━━━━━━━━━━━━
📂 目录: {directory_path}
📊 进度: [{progress_bar}] {current_num}/{total} ({progress_percent}%)
✅ 成功: {success_count} 个
❌ 失败: {fail_count} 个
💰 总Token: {total_tokens:,}
"""

        yield result_summary, df

    # 最终完成总结
    final_summary = f"""
🎉 批量分析全部完成！
━━━━━━━━━━━━━━━━━━━━
📂 目录: {directory_path}
📊 总数: {total} 个视频
✅ 成功: {success_count} 个
❌ 失败: {fail_count} 个
💰 总Token: {total_tokens:,}
━━━━━━━━━━━━━━━━━━━━
"""

    yield final_summary, df


def rename_single_video(row_index):
    """重命名单个视频"""
    global video_analysis_results

    if not video_analysis_results:
        return "❌ 请先分析视频", None

    video_paths = list(video_analysis_results.keys())
    if row_index < 0 or row_index >= len(video_paths):
        return "❌ 无效的行索引", None

    video_path = video_paths[row_index]
    info = video_analysis_results[video_path]

    try:
        old_path = Path(info["original_path"])

        if not old_path.exists():
            return f"❌ 文件不存在: {old_path}", None

        description = sanitize_filename(info["description"])

        if not description:
            return "❌ 描述内容为空，无法重命名", None

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

        # 更新存储
        del video_analysis_results[video_path]
        video_analysis_results[str(new_path)] = {
            **info,
            "original_name": new_name,
            "original_path": str(new_path)
        }

        msg = f"""✅ 重命名成功！

原文件名:
{old_path.name}

新文件名:
{new_name}

完整路径:
{new_path}
"""

        # 更新表格
        df = generate_current_dataframe()
        return msg, df

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f"❌ 重命名失败: {str(e)}\n\n{error_detail}", None


def rename_all_videos():
    """批量重命名所有视频"""
    global video_analysis_results

    if not video_analysis_results:
        return "❌ 请先分析视频", None

    success_count = 0
    fail_count = 0
    messages = []

    # 创建副本避免迭代时修改
    items = list(video_analysis_results.items())

    for video_path, info in items:
        try:
            old_path = Path(info["original_path"])

            if not old_path.exists():
                fail_count += 1
                messages.append(f"❌ {old_path.name} - 文件不存在")
                continue

            description = sanitize_filename(info["description"])

            if not description:
                fail_count += 1
                messages.append(f"❌ {old_path.name} - 描述为空")
                continue

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
                del video_analysis_results[video_path]
                video_analysis_results[str(new_path)] = {
                    **info,
                    "original_name": new_name,
                    "original_path": str(new_path)
                }

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
""" + "\n".join(messages[:50])  # 限制显示前50条

    if len(messages) > 50:
        summary += f"\n\n... 还有 {len(messages) - 50} 条记录"

    df = generate_current_dataframe()
    return summary, df


def generate_current_dataframe():
    """生成当前的DataFrame"""
    global video_analysis_results

    results = []
    for idx, (video_path, info) in enumerate(video_analysis_results.items(), 1):
        file_size = Path(info["original_path"]).stat().st_size / 1024 / 1024 if Path(info["original_path"]).exists() else 0
        results.append({
            "序号": idx,
            "文件名": info["original_name"],
            "大小(MB)": f"{file_size:.2f}",
            "视频描述": info["description"],
            "Token": info["stats"]["total_tokens"],
            "状态": "✅ 已分析"
        })

    return pd.DataFrame(results)


def batch_analyze_storyboards(directory_path, model_name, max_tokens, enable_thinking, prompt):
    """批量分析目录中视频的分镜脚本 - 流式展示结果"""
    global storyboard_results
    storyboard_results.clear()

    if not directory_path or not os.path.exists(directory_path):
        yield "❌ 请输入有效的目录路径", None, ""
        return

    video_files = get_video_files(directory_path)

    if not video_files:
        yield "❌ 目录中没有视频文件", None, ""
        return

    results = []
    total = len(video_files)
    total_tokens = 0
    success_count = 0
    fail_count = 0

    # 用于生成详细展示
    detailed_output = []

    # 初始状态
    initial_summary = f"""
🚀 开始批量分镜脚本解析
━━━━━━━━━━━━━━━━━━━━
📂 目录: {directory_path}
📊 总数: {total} 个视频
⏳ 准备开始处理...
"""
    yield initial_summary, None, ""

    for idx, video_path in enumerate(video_files):
        video_name = Path(video_path).name

        # 检查文件是否存在
        if not Path(video_path).exists():
            # 文件不存在，跳过
            error_msg = "文件不存在"
            detailed_output.append(f"\n{'='*80}")
            detailed_output.append(f"❌ 视频: {video_name}")
            detailed_output.append(f"{'='*80}")
            detailed_output.append(f"错误: {error_msg}\n")

            results.append({
                "序号": idx + 1,
                "视频名称": video_name,
                "大小(MB)": "N/A",
                "分镜数": 0,
                "Token": 0,
                "状态": f"❌ 失败: {error_msg}"
            })
            fail_count += 1
            continue

        file_size = Path(video_path).stat().st_size / 1024 / 1024

        # 处理前显示状态
        current_num = idx + 1
        progress_percent = int(current_num / total * 100)
        progress_bar = "█" * (progress_percent // 5) + "░" * (20 - progress_percent // 5)

        # 添加loading动画效果
        loading_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        loading_icon = loading_frames[idx % len(loading_frames)]

        processing_summary = f"""
{loading_icon} 正在解析分镜脚本 {current_num}/{total}
━━━━━━━━━━━━━━━━━━━━
📂 目录: {directory_path}
📊 进度: [{progress_bar}] {current_num}/{total} ({progress_percent}%)
📹 当前: {video_name} ({file_size:.1f}MB)
⏳ 解析中，请稍候...
"""
        yield processing_summary, pd.DataFrame(results) if results else None, "\n".join(detailed_output)

        success, storyboard, stats = analyze_storyboard(
            video_path, model_name, max_tokens, enable_thinking, prompt
        )

        if success and storyboard:
            # 存储结果
            storyboard_results[video_path] = {
                "video_name": video_name,
                "storyboard": storyboard,
                "stats": stats,
            }

            scene_count = len(storyboard)
            tokens = stats["total_tokens"]
            total_tokens += tokens

            results.append({
                "序号": idx + 1,
                "视频名称": video_name,
                "大小(MB)": f"{file_size:.2f}",
                "分镜数": scene_count,
                "Token": tokens,
                "状态": "✅ 成功"
            })

            success_count += 1

            # 构建详细展示
            detailed_output.append(f"\n{'='*80}")
            detailed_output.append(f"📹 视频: {video_name}")
            detailed_output.append(f"{'='*80}\n")

            # 创建分镜表格
            storyboard_df = []
            for scene in storyboard:
                storyboard_df.append({
                    "分镜号": scene.get("boardNo", ""),
                    "口播字幕": scene.get("subtitle", "").replace("\\", "\n"),
                    "说话人": scene.get("speaker", ""),
                    "是否正打": "是" if scene.get("straightOn", False) else "侧打/其他",
                    "是否浮屏": "是" if scene.get("floatingScreen", False) else "否",
                    "画面描述": scene.get("scene", "")
                })

            df = pd.DataFrame(storyboard_df)
            detailed_output.append(df.to_string(index=False))
            detailed_output.append("\n")

        else:
            error_msg = stats.get("error", "未知错误")
            # 添加详细的错误信息到输出
            detailed_output.append(f"\n{'='*80}")
            detailed_output.append(f"❌ 视频: {video_name}")
            detailed_output.append(f"{'='*80}")
            detailed_output.append(f"错误: {error_msg}")
            if "detail" in stats:
                detailed_output.append(f"\n详细信息:\n{stats['detail']}")
            detailed_output.append("\n")

            results.append({
                "序号": idx + 1,
                "视频名称": video_name,
                "大小(MB)": f"{file_size:.2f}",
                "分镜数": 0,
                "Token": 0,
                "状态": f"❌ 失败: {error_msg}"
            })
            fail_count += 1

        # 分析完成后立即更新结果
        df = pd.DataFrame(results)

        result_summary = f"""
✅ 已完成 {current_num}/{total} 个视频
━━━━━━━━━━━━━━━━━━━━
📂 目录: {directory_path}
📊 进度: [{progress_bar}] {current_num}/{total} ({progress_percent}%)
✅ 成功: {success_count} 个
❌ 失败: {fail_count} 个
💰 总Token: {total_tokens:,}
"""

        detailed_text = "\n".join(detailed_output)

        yield result_summary, df, detailed_text

    # 最终完成总结
    final_summary = f"""
🎉 批量分镜脚本解析全部完成！
━━━━━━━━━━━━━━━━━━━━
📂 目录: {directory_path}
📊 总数: {total} 个视频
✅ 成功: {success_count} 个
❌ 失败: {fail_count} 个
💰 总Token: {total_tokens:,}
━━━━━━━━━━━━━━━━━━━━
"""

    yield final_summary, df, detailed_text


def export_storyboards_to_excel(output_path: str = None):
    """导出所有分镜脚本到Excel文件"""
    global storyboard_results

    if not storyboard_results:
        return "❌ 没有可导出的分镜脚本数据"

    # 确定输出路径
    if not output_path or not output_path.strip():
        # 使用Downloads目录作为默认目录
        import os
        downloads_path = "/Users/chen/Downloads"
        output_path = os.path.join(downloads_path, "分镜脚本汇总.xlsx")
    else:
        # 检查用户输入的路径
        output_path_obj = Path(output_path)

        # 如果是已存在的目录,在其中创建默认文件名
        if output_path_obj.exists() and output_path_obj.is_dir():
            output_path = output_path_obj / "分镜脚本汇总.xlsx"
        # 如果路径不存在,但看起来像目录(没有扩展名)
        elif not output_path_obj.suffix:
            # 当作目录处理,添加默认文件名
            output_path = output_path_obj / "分镜脚本汇总.xlsx"
        # 如果有文件名但没有.xlsx扩展名,添加扩展名
        elif not str(output_path).endswith('.xlsx'):
            output_path = str(output_path) + '.xlsx'

        output_path = str(output_path)

    # 确保目标目录存在
    output_path_obj = Path(output_path)
    parent_dir = output_path_obj.parent

    try:
        # 创建目录(如果不存在)
        parent_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return f"❌ 无法创建目录: {parent_dir}\n错误: {str(e)}"

    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for video_path, info in storyboard_results.items():
                video_name = info["video_name"]
                storyboard = info["storyboard"]

                # 创建分镜表格
                df_data = []
                for scene in storyboard:
                    df_data.append({
                        "分镜号": scene.get("boardNo", ""),
                        "口播字幕": scene.get("subtitle", "").replace("\\", "\n"),
                        "说话人": scene.get("speaker", ""),
                        "是否正打": "正打" if scene.get("straightOn", False) else "否",
                        "是否浮屏": "浮屏" if scene.get("floatingScreen", False) else "否",
                        "画面描述": scene.get("scene", "")
                    })

                df = pd.DataFrame(df_data)

                # 使用视频名作为sheet名（处理非法字符）
                sheet_name = re.sub(r'[<>:"/\\|?*\[\]]', '_', video_name)[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        return f"✅ 导出成功！\n文件路径: {output_path}"

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f"❌ 导出失败: {str(e)}\n\n{error_detail}"


def analyze_video_prompt(video_path: str, model_name: str, max_tokens: int, enable_thinking: bool = False, prompt: str = None) -> Tuple[bool, str, dict]:
    """分析视频并生成AI视频提示词"""
    try:
        # 使用传入的提示词或默认提示词
        if not prompt:
            prompt = VIDEO_PROMPT_REVERSE_PROMPT

        # 读取视频
        path = Path(video_path)
        file_size = path.stat().st_size

        # 文件大小限制 (50MB)
        if file_size > 50 * 1024 * 1024:
            return False, "文件过大（>50MB）", {"error": "文件过大（>50MB）"}

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

        # 根据思考模式设置
        if not enable_thinking:
            request_body["systemInstruction"] = {
                "parts": [{"text": "请直接给出详细的视频提示词分析，不要展示思考过程。"}]
            }

        # 调用API
        response = requests.post(
            f"{API_BASE}/gemini/v1beta/models/{model_name}:generateContent",
            headers={
                "x-goog-api-key": API_KEY,
                "Content-Type": "application/json"
            },
            json=request_body,
            timeout=300  # 提示词反推可能需要较长时间
        )

        if response.status_code != 200:
            return False, f"API错误: {response.status_code}", {"error": f"API错误: {response.status_code}"}

        result = response.json()

        # 检查是否有内容
        if not result.get("candidates") or not result["candidates"][0].get("content"):
            return False, "API返回空内容", {"error": "API返回空内容"}

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
        import traceback
        error_detail = traceback.format_exc()
        return False, f"错误: {str(e)}", {"error": f"错误: {str(e)}", "detail": error_detail}


def batch_analyze_video_prompts(directory_path, model_name, max_tokens, enable_thinking, prompt):
    """批量分析目录中视频的AI提示词 - 流式展示结果"""
    global video_prompt_results
    video_prompt_results.clear()

    if not directory_path or not os.path.exists(directory_path):
        yield "❌ 请输入有效的目录路径", None, ""
        return

    video_files = get_video_files(directory_path)

    if not video_files:
        yield "❌ 目录中没有视频文件", None, ""
        return

    results = []
    total = len(video_files)
    total_tokens = 0
    success_count = 0
    fail_count = 0

    # 初始状态
    initial_summary = f"""
🚀 开始批量视频提示词反推
━━━━━━━━━━━━━━━━━━━━
📂 目录: {directory_path}
📊 总数: {total} 个视频
⏳ 准备开始处理...
"""
    yield initial_summary, None, "📌 提示：分析完成后，所有视频的提示词将在此处平铺展示，无需手动选择"

    for idx, video_path in enumerate(video_files):
        video_name = Path(video_path).name

        # 检查文件是否存在
        if not Path(video_path).exists():
            # 文件不存在，跳过
            results.append({
                "序号": idx + 1,
                "视频名称": video_name,
                "大小(MB)": "N/A",
                "Token": 0,
                "状态": "❌ 失败: 文件不存在"
            })
            fail_count += 1
            continue

        file_size = Path(video_path).stat().st_size / 1024 / 1024

        # 处理前显示状态
        current_num = idx + 1
        progress_percent = int(current_num / total * 100)
        progress_bar = "█" * (progress_percent // 5) + "░" * (20 - progress_percent // 5)

        # 添加loading动画效果
        loading_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        loading_icon = loading_frames[idx % len(loading_frames)]

        processing_summary = f"""
{loading_icon} 正在反推提示词 {current_num}/{total}
━━━━━━━━━━━━━━━━━━━━
📂 目录: {directory_path}
📊 进度: [{progress_bar}] {current_num}/{total} ({progress_percent}%)
📹 当前: {video_name} ({file_size:.1f}MB)
⏳ 反推中，请稍候...
"""
        yield processing_summary, pd.DataFrame(results) if results else None, generate_all_prompts_display()

        success, prompt_content, stats = analyze_video_prompt(
            video_path, model_name, max_tokens, enable_thinking, prompt
        )

        if success:
            # 存储结果
            video_prompt_results[video_path] = {
                "video_name": video_name,
                "prompt_content": prompt_content,
                "stats": stats,
            }

            tokens = stats["total_tokens"]
            total_tokens += tokens

            results.append({
                "序号": idx + 1,
                "视频名称": video_name,
                "大小(MB)": f"{file_size:.2f}",
                "Token": tokens,
                "状态": "✅ 成功"
            })

            success_count += 1

        else:
            error_msg = stats.get("error", "未知错误")
            results.append({
                "序号": idx + 1,
                "视频名称": video_name,
                "大小(MB)": f"{file_size:.2f}",
                "Token": 0,
                "状态": f"❌ 失败: {error_msg}"
            })
            fail_count += 1

        # 分析完成后立即更新结果
        df = pd.DataFrame(results)

        result_summary = f"""
✅ 已完成 {current_num}/{total} 个视频
━━━━━━━━━━━━━━━━━━━━
📂 目录: {directory_path}
📊 进度: [{progress_bar}] {current_num}/{total} ({progress_percent}%)
✅ 成功: {success_count} 个
❌ 失败: {fail_count} 个
💰 总Token: {total_tokens:,}
"""

        # 实时生成已分析视频的提示词展示
        all_prompts_display = generate_all_prompts_display()

        yield result_summary, df, all_prompts_display

    # 最终完成总结
    final_summary = f"""
🎉 批量视频提示词反推全部完成！
━━━━━━━━━━━━━━━━━━━━
📂 目录: {directory_path}
📊 总数: {total} 个视频
✅ 成功: {success_count} 个
❌ 失败: {fail_count} 个
💰 总Token: {total_tokens:,}
━━━━━━━━━━━━━━━━━━━━
"""

    yield final_summary, df, all_prompts_display


def generate_all_prompts_display() -> str:
    """生成所有视频提示词的平铺展示"""
    global video_prompt_results

    if not video_prompt_results:
        return "📌 提示：分析完成后，所有视频的提示词将在此处平铺展示，无需手动选择"

    output_parts = []

    for idx, (video_path, info) in enumerate(video_prompt_results.items(), 1):
        output_parts.append(f"""
{'='*100}
📹 视频 {idx}: {info['video_name']}
{'='*100}

{info['prompt_content']}

{'─'*100}
📊 统计信息: Token使用: {info['stats']['total_tokens']:,} | 输入Token: {info['stats']['prompt_tokens']:,} | 输出Token: {info['stats']['output_tokens']:,} | 完成原因: {info['stats']['finish_reason']}
{'='*100}
""")

    return "\n".join(output_parts)


def get_video_prompt_detail(video_index: int) -> str:
    """获取指定视频的详细提示词内容（保留向后兼容）"""
    global video_prompt_results

    if not video_prompt_results:
        return "❌ 请先分析视频"

    video_paths = list(video_prompt_results.keys())
    if video_index < 0 or video_index >= len(video_paths):
        return "❌ 无效的视频序号"

    video_path = video_paths[video_index]
    info = video_prompt_results[video_path]

    output = f"""
{'='*80}
📹 视频: {info['video_name']}
{'='*80}

{info['prompt_content']}

{'='*80}
📊 统计信息
Token使用: {info['stats']['total_tokens']:,}
输入Token: {info['stats']['prompt_tokens']:,}
输出Token: {info['stats']['output_tokens']:,}
完成原因: {info['stats']['finish_reason']}
{'='*80}
"""
    return output


def export_video_prompts_to_markdown(output_path: str = None):
    """导出所有视频提示词到Markdown文件"""
    global video_prompt_results

    if not video_prompt_results:
        return "❌ 没有可导出的视频提示词数据"

    # 确定输出路径
    if not output_path or not output_path.strip():
        import os
        downloads_path = "/Users/chen/Downloads"
        output_path = os.path.join(downloads_path, "视频提示词汇总.md")
    else:
        output_path_obj = Path(output_path)

        if output_path_obj.exists() and output_path_obj.is_dir():
            output_path = output_path_obj / "视频提示词汇总.md"
        elif not output_path_obj.suffix:
            output_path = output_path_obj / "视频提示词汇总.md"
        elif not str(output_path).endswith('.md'):
            output_path = str(output_path) + '.md'

        output_path = str(output_path)

    # 确保目标目录存在
    output_path_obj = Path(output_path)
    parent_dir = output_path_obj.parent

    try:
        parent_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return f"❌ 无法创建目录: {parent_dir}\n错误: {str(e)}"

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 视频AI提示词反推汇总\n\n")
            f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"视频总数: {len(video_prompt_results)}\n\n")
            f.write("---\n\n")

            for idx, (video_path, info) in enumerate(video_prompt_results.items(), 1):
                f.write(f"## {idx}. {info['video_name']}\n\n")
                f.write(f"{info['prompt_content']}\n\n")
                f.write(f"**统计信息:**\n")
                f.write(f"- Token使用: {info['stats']['total_tokens']:,}\n")
                f.write(f"- 输入Token: {info['stats']['prompt_tokens']:,}\n")
                f.write(f"- 输出Token: {info['stats']['output_tokens']:,}\n\n")
                f.write("---\n\n")

        return f"✅ 导出成功！\n文件路径: {output_path}"

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f"❌ 导出失败: {str(e)}\n\n{error_detail}"


def export_video_prompts_to_excel(output_path: str = None):
    """导出所有视频提示词到Excel文件"""
    global video_prompt_results

    if not video_prompt_results:
        return "❌ 没有可导出的视频提示词数据"

    # 确定输出路径
    if not output_path or not output_path.strip():
        import os
        downloads_path = "/Users/chen/Downloads"
        output_path = os.path.join(downloads_path, "视频提示词汇总.xlsx")
    else:
        output_path_obj = Path(output_path)

        if output_path_obj.exists() and output_path_obj.is_dir():
            output_path = output_path_obj / "视频提示词汇总.xlsx"
        elif not output_path_obj.suffix:
            output_path = output_path_obj / "视频提示词汇总.xlsx"
        elif not str(output_path).endswith('.xlsx'):
            output_path = str(output_path) + '.xlsx'

        output_path = str(output_path)

    # 确保目标目录存在
    output_path_obj = Path(output_path)
    parent_dir = output_path_obj.parent

    try:
        parent_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return f"❌ 无法创建目录: {parent_dir}\n错误: {str(e)}"

    try:
        df_data = []
        for idx, (video_path, info) in enumerate(video_prompt_results.items(), 1):
            df_data.append({
                "序号": idx,
                "视频名称": info["video_name"],
                "视频提示词": info["prompt_content"],
                "总Token": info["stats"]["total_tokens"],
                "输入Token": info["stats"]["prompt_tokens"],
                "输出Token": info["stats"]["output_tokens"],
                "完成状态": info["stats"]["finish_reason"]
            })

        df = pd.DataFrame(df_data)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name="视频提示词汇总", index=False)

            # 调整列宽
            worksheet = writer.sheets["视频提示词汇总"]
            worksheet.column_dimensions['A'].width = 8
            worksheet.column_dimensions['B'].width = 40
            worksheet.column_dimensions['C'].width = 100
            worksheet.column_dimensions['D'].width = 12
            worksheet.column_dimensions['E'].width = 12
            worksheet.column_dimensions['F'].width = 12
            worksheet.column_dimensions['G'].width = 15

        return f"✅ 导出成功！\n文件路径: {output_path}"

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f"❌ 导出失败: {str(e)}\n\n{error_detail}"


# 创建Gradio界面
with gr.Blocks(title="批量视频分析工具 V2 + 分镜脚本 + 提示词反推", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🎬 批量视频分析工具 V2 + 分镜脚本 + 提示词反推
    ### 基于 Gemini Balance API - 本地目录批量处理
    **三大功能**: 视频描述与重命名 | 分镜脚本解析 | AI提示词反推
    """)

    # 共享的目录选择区域
    gr.Markdown("### 📂 选择视频目录")
    gr.Markdown("💡 **提示**: 在Finder中右键目录 → 按住Option键 → 点击\"将xxx拷贝为路径名称\"，然后粘贴到下方")
    directory_input = gr.Textbox(
        label="视频目录路径",
        value=DEFAULT_VIDEO_DIR,
        placeholder="例如: /Users/yourname/Videos",
        lines=1
    )
    scan_btn = gr.Button("🔍 扫描目录", variant="secondary")

    # 使用Tab分隔两个功能
    with gr.Tabs():
        # Tab 1: 视频描述与重命名
        with gr.Tab("📝 视频描述与重命名"):
            with gr.Row():
                # 左侧：配置面板
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ 分析配置")
                    model_selector = gr.Dropdown(
                        choices=AVAILABLE_MODELS,
                        value=AVAILABLE_MODELS[0],
                        label="Gemini模型"
                    )

                    max_tokens = gr.Slider(
                        minimum=2048,
                        maximum=8192,
                        value=4096,
                        step=512,
                        label="最大输出Token",
                        info="建议4096以上，思考模式会消耗大量Token"
                    )

                    enable_thinking_checkbox = gr.Checkbox(
                        label="启用思考模式",
                        value=False,
                        info="⚠️ 思考模式会消耗1000-2000个额外Token，可能导致内容为空。推荐关闭。"
                    )

                    prompt_input = gr.Textbox(
                        label="提示词",
                        value=DEFAULT_PROMPT,
                        lines=6
                    )

                    analyze_btn = gr.Button("🚀 开始批量分析", variant="primary", size="lg")

                # 右侧：结果面板
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 分析结果")
                    status_output = gr.Textbox(
                        label="状态信息",
                        lines=8,
                        show_copy_button=True
                    )

                    results_table = gr.Dataframe(
                        label="视频列表",
                        wrap=True,
                        interactive=False
                    )

            # 重命名操作区
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🔄 单个重命名")
                    with gr.Row():
                        row_index_input = gr.Number(
                            label="视频序号",
                            value=1,
                            minimum=1,
                            precision=0
                        )
                        rename_single_btn = gr.Button("重命名", variant="secondary")

                with gr.Column(scale=2):
                    gr.Markdown("### 🔄 批量重命名")
                    rename_all_btn = gr.Button("🔄 重命名全部视频", variant="primary", size="lg")

            rename_output = gr.Textbox(
                label="重命名结果",
                lines=10,
                show_copy_button=True
            )

        # Tab 2: 分镜脚本解析
        with gr.Tab("🎞️ 分镜脚本解析"):
            with gr.Row():
                # 左侧：配置面板
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ 分镜解析配置")

                    storyboard_model_selector = gr.Dropdown(
                        choices=AVAILABLE_MODELS,
                        value=AVAILABLE_MODELS[0],
                        label="Gemini模型"
                    )

                    storyboard_max_tokens = gr.Slider(
                        minimum=8192,
                        maximum=100000,
                        value=100000,
                        step=1024,
                        label="最大输出Token",
                        info="分镜解析需要大量Token，默认10万确保完整输出"
                    )

                    storyboard_enable_thinking = gr.Checkbox(
                        label="启用思考模式",
                        value=False,
                        info="⚠️ 分镜解析推荐关闭思考模式"
                    )

                    storyboard_prompt_input = gr.Textbox(
                        label="分镜解析提示词",
                        value=STORYBOARD_PROMPT,
                        lines=10,
                        info="可根据需要自定义提示词"
                    )

                    storyboard_analyze_btn = gr.Button("🎬 开始分镜脚本解析", variant="primary", size="lg")

                # 右侧：结果面板
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 解析结果")
                    storyboard_status_output = gr.Textbox(
                        label="状态信息",
                        lines=8,
                        show_copy_button=True
                    )

                    storyboard_results_table = gr.Dataframe(
                        label="视频分镜统计",
                        wrap=True,
                        interactive=False
                    )

            # 详细分镜展示
            gr.Markdown("### 📋 详细分镜脚本")
            storyboard_detail_output = gr.Textbox(
                label="分镜脚本详情（每个视频一个表格）",
                lines=30,
                show_copy_button=True
            )

            # 导出功能
            export_path_input = gr.Textbox(
                label="导出文件路径（可选）",
                placeholder="留空则默认保存到 /Users/chen/Downloads/分镜脚本汇总.xlsx",
                lines=1,
                info="支持自定义路径，例如: /Users/chen/Downloads/分镜脚本.xlsx"
            )
            export_btn = gr.Button("📤 导出到Excel", variant="primary")

            export_output = gr.Textbox(
                label="导出结果",
                lines=3,
                show_copy_button=True
            )

        # Tab 3: 视频提示词反推
        with gr.Tab("🎨 视频提示词反推"):
            with gr.Row():
                # 左侧：配置面板
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ 提示词反推配置")

                    prompt_model_selector = gr.Dropdown(
                        choices=AVAILABLE_MODELS,
                        value=AVAILABLE_MODELS[1],  # 默认使用pro模型
                        label="Gemini模型",
                        info="推荐使用gemini-2.5-pro获得更专业的分析"
                    )

                    prompt_max_tokens = gr.Slider(
                        minimum=8192,
                        maximum=100000,
                        value=16384,
                        step=1024,
                        label="最大输出Token",
                        info="提示词反推需要较多Token，建议16384"
                    )

                    prompt_enable_thinking = gr.Checkbox(
                        label="启用思考模式",
                        value=True,  # Pro模型默认启用
                        interactive=False,  # Pro模型不允许关闭
                        info="✓ Pro模型默认启用思考模式且无法关闭"
                    )

                    prompt_input_box = gr.Textbox(
                        label="提示词反推指令（高级用户可自定义）",
                        value=VIDEO_PROMPT_REVERSE_PROMPT,
                        lines=15,
                        info="可根据需要自定义提示词"
                    )

                    prompt_analyze_btn = gr.Button("🎨 开始提示词反推", variant="primary", size="lg")

                # 右侧：结果面板
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 反推结果")
                    prompt_status_output = gr.Textbox(
                        label="状态信息",
                        lines=8,
                        show_copy_button=True
                    )

                    prompt_results_table = gr.Dataframe(
                        label="视频提示词统计",
                        wrap=True,
                        interactive=False
                    )

            # 详细提示词展示 - 平铺展示所有内容
            gr.Markdown("### 📋 详细提示词内容（所有视频）")

            prompt_detail_output = gr.Textbox(
                value="📌 提示：分析完成后，所有视频的提示词将在此处平铺展示，无需手动选择",
                lines=30,
                max_lines=50,
                show_copy_button=True,
                interactive=False,
                container=True
            )

            # 导出功能
            gr.Markdown("### 📤 导出功能")
            prompt_export_path_input = gr.Textbox(
                label="导出文件路径（可选）",
                placeholder="留空则默认保存到 /Users/chen/Downloads/",
                lines=1,
                info="支持自定义路径，例如: /Users/chen/Downloads/视频提示词.xlsx 或 .md"
            )

            with gr.Row():
                prompt_export_md_btn = gr.Button("📄 导出为Markdown", variant="primary", scale=1)
                prompt_export_excel_btn = gr.Button("📊 导出为Excel", variant="primary", scale=1)

            prompt_export_output = gr.Textbox(
                label="导出结果",
                lines=3,
                show_copy_button=True
            )

    # 使用说明
    gr.Markdown("""
    ---
    ### 💡 使用说明

    #### 📝 视频描述与重命名功能
    1. **扫描目录**: 输入视频目录路径，点击"🔍 扫描目录"
    2. **批量分析**: 选择模型和配置，点击"🚀 开始批量分析"
    3. **重命名文件**: 分析完成后可单个或批量重命名视频

    #### 🎞️ 分镜脚本解析功能
    1. **扫描目录**: 先扫描视频目录
    2. **分镜解析**: 切换到"分镜脚本解析"标签页
    3. **配置模型**: 选择模型（推荐 `gemini-2.5-flash`），Token默认10万
    4. **开始解析**: 点击"🎬 开始分镜脚本解析"
    5. **查看结果**: 在"详细分镜脚本"区域查看每个视频的分镜表格
    6. **导出Excel**: 点击"📤 导出到Excel"保存所有分镜脚本

    #### 🎨 视频提示词反推功能（新增）
    1. **扫描目录**: 先扫描视频目录
    2. **提示词反推**: 切换到"视频提示词反推"标签页
    3. **配置模型**: 推荐使用 `gemini-2.5-pro` 模型，Token默认16384
    4. **开始反推**: 点击"🎨 开始提示词反推"
    5. **查看详情**: 选择视频序号，点击"👁️ 查看详情"查看完整提示词
    6. **导出结果**: 支持导出为Markdown或Excel格式

    #### 📋 字段说明
    **分镜脚本字段：**
    - **分镜号**: 从1开始的连续编号
    - **口播字幕**: 视频中出现的字幕内容
    - **说话人**: 说话人的名称或身份
    - **是否正打**: 人物是否全屏正对镜头（浮窗不算正打）
    - **是否浮屏**: 是否为人物浮窗/画中画形式
    - **画面描述**: 详细的镜头画面描述

    **视频提示词字段：**
    - **英文提示词**: 包含概念、场景、人物、动作、摄影、音频等维度
    - **中文释义**: 对应的中文翻译，便于理解

    #### 📌 重要提示
    - ✅ 三个功能互相独立，可分别使用
    - ⚠️ 分镜解析消耗Token较多，默认10万Token确保完整输出
    - 🎨 提示词反推使用Pro模型效果更好，消耗Token约8k-16k
    - 🎯 分镜解析推荐 gemini-2.5-flash，提示词反推推荐 gemini-2.5-pro
    - 🔕 建议关闭思考模式，避免额外Token消耗
    - 💾 重命名操作不可撤销，请谨慎使用
    - 📊 所有功能都支持导出，默认保存到桌面
    - 💻 导出路径可自定义
    - 📁 目标目录不存在会自动创建
    """)

    # 绑定事件 - Tab 1: 视频描述与重命名
    model_selector.change(
        fn=update_thinking_mode_state,
        inputs=[model_selector],
        outputs=[enable_thinking_checkbox]
    )

    scan_btn.click(
        fn=scan_directory,
        inputs=[directory_input],
        outputs=[status_output, results_table]
    )

    analyze_btn.click(
        fn=batch_analyze_videos,
        inputs=[directory_input, model_selector, prompt_input, max_tokens, enable_thinking_checkbox],
        outputs=[status_output, results_table]
    )

    rename_single_btn.click(
        fn=lambda idx: rename_single_video(int(idx) - 1),
        inputs=[row_index_input],
        outputs=[rename_output, results_table]
    )

    rename_all_btn.click(
        fn=rename_all_videos,
        outputs=[rename_output, results_table]
    )

    # 绑定事件 - Tab 2: 分镜脚本解析
    storyboard_model_selector.change(
        fn=update_thinking_mode_state,
        inputs=[storyboard_model_selector],
        outputs=[storyboard_enable_thinking]
    )

    storyboard_analyze_btn.click(
        fn=batch_analyze_storyboards,
        inputs=[directory_input, storyboard_model_selector, storyboard_max_tokens, storyboard_enable_thinking, storyboard_prompt_input],
        outputs=[storyboard_status_output, storyboard_results_table, storyboard_detail_output]
    )

    export_btn.click(
        fn=lambda path: export_storyboards_to_excel(path if path.strip() else None),
        inputs=[export_path_input],
        outputs=[export_output]
    )

    # 绑定事件 - Tab 3: 视频提示词反推
    prompt_model_selector.change(
        fn=update_thinking_mode_state,
        inputs=[prompt_model_selector],
        outputs=[prompt_enable_thinking]
    )

    prompt_analyze_btn.click(
        fn=batch_analyze_video_prompts,
        inputs=[directory_input, prompt_model_selector, prompt_max_tokens, prompt_enable_thinking, prompt_input_box],
        outputs=[prompt_status_output, prompt_results_table, prompt_detail_output]
    )

    prompt_export_md_btn.click(
        fn=lambda path: export_video_prompts_to_markdown(path if path.strip() else None),
        inputs=[prompt_export_path_input],
        outputs=[prompt_export_output]
    )

    prompt_export_excel_btn.click(
        fn=lambda path: export_video_prompts_to_excel(path if path.strip() else None),
        inputs=[prompt_export_path_input],
        outputs=[prompt_export_output]
    )


if __name__ == "__main__":
    print("=" * 80)
    print("🎬 批量视频分析与重命名工具 V2 + 分镜脚本 + 提示词反推")
    print("=" * 80)
    print(f"\n📡 API地址: {API_BASE}")
    print(f"📂 默认目录: {DEFAULT_VIDEO_DIR}")
    print(f"\n🌐 启动Web界面...")
    print("\n✨ 功能列表:")
    print("  ✓ 视频描述与重命名")
    print("  ✓ 分镜脚本解析")
    print("  ✓ 视频提示词反推 (新增)")
    print("  ✓ 批量处理本地视频目录")
    print("  ✓ 多格式导出(Excel/Markdown)")
    print("\n📋 功能说明:")
    print("  • 视频描述: 生成50字以内的视频描述，可用于重命名")
    print("  • 分镜脚本: 解析视频分镜，包含字幕、说话人、正打浮屏等信息")
    print("  • 提示词反推: 生成AI视频模型(Sora/Runway/Pika)可用的详细提示词")
    print("\n⚠️  重命名操作不可撤销，请谨慎使用！")
    print("\n提示: 按 Ctrl+C 退出\n")

    app.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        inbrowser=True
    )
