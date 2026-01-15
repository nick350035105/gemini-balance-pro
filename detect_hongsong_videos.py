#!/usr/bin/env python3
"""
检测视频第一帧是否包含"红松"文字或logo，并重命名文件
"""
import cv2
import requests
import base64
import json
from pathlib import Path
from typing import List, Tuple
import time

# 配置
API_BASE = "http://localhost:8000"
API_KEY = "sk-demo-token"
MODEL = "gemini-2.5-flash"
VIDEO_DIR = "/Users/chen/ClaudeCode/工作项目/行业素材库/乐器行业/视频数据2"

# 检测提示词
DETECTION_PROMPT = """请仔细查看这张图片，检测是否包含"红松"这两个字或者红松品牌的logo。

要求：
1. 如果图片中包含"红松"文字（简体或繁体），回答：是
2. 如果图片中包含红松品牌的logo或标识，回答：是
3. 如果都不包含，回答：否

只需要回答"是"或"否"，不要有其他内容。"""


def extract_first_frame(video_path: str) -> bytes:
    """提取视频第一帧并返回JPG格式的字节数据"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception(f"无法打开视频: {video_path}")

    # 读取第一帧
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise Exception(f"无法读取视频帧: {video_path}")

    # 转为JPG格式
    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        raise Exception(f"无法编码图片: {video_path}")

    return buffer.tobytes()


def detect_hongsong_in_image(image_data: bytes) -> Tuple[bool, str]:
    """
    使用Gemini API检测图片中是否包含"红松"
    返回: (是否包含, API响应文本)
    """
    try:
        # 转为base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # 构建请求
        request_body = {
            "contents": [{
                "parts": [
                    {"text": DETECTION_PROMPT},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_base64
                        }
                    }
                ]
            }],
            "systemInstruction": {
                "parts": [{"text": "直接回答问题，不要思考过程。"}]
            },
            "generationConfig": {
                "temperature": 0.1,  # 降低温度以获得更确定的结果
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 500,  # 增加到500以适应思考模式
                "responseModalities": ["TEXT"],  # 只需要文本响应
            }
        }

        # 调用API
        response = requests.post(
            f"{API_BASE}/gemini/v1beta/models/{MODEL}:generateContent",
            headers={
                "x-goog-api-key": API_KEY,
                "Content-Type": "application/json"
            },
            json=request_body,
            timeout=30
        )

        if response.status_code != 200:
            return False, f"API错误: {response.status_code}"

        result = response.json()
        text_content = result["candidates"][0]["content"]["parts"][0]["text"].strip()

        # 判断是否包含"红松"
        contains_hongsong = "是" in text_content

        return contains_hongsong, text_content

    except Exception as e:
        return False, f"错误: {str(e)}"


def process_videos(video_dir: str):
    """处理目录中的所有视频"""
    print("=" * 80)
    print("🎬 红松视频检测与重命名工具")
    print("=" * 80)

    # 获取所有视频文件
    video_dir_path = Path(video_dir)
    video_files = sorted(video_dir_path.glob("*.mp4"))

    total = len(video_files)
    print(f"\n📂 目录: {video_dir}")
    print(f"📊 视频总数: {total}")
    print(f"🤖 使用模型: {MODEL}")
    print("\n" + "=" * 80)

    # 统计
    processed = 0
    detected = 0
    renamed = 0
    errors = 0

    # 处理每个视频
    for idx, video_file in enumerate(video_files, 1):
        video_name = video_file.name

        # 检查是否已经重命名过
        if video_name.startswith("红松_"):
            print(f"\n[{idx}/{total}] ⏭️  已重命名: {video_name}")
            continue

        print(f"\n[{idx}/{total}] 📹 处理: {video_name}")

        try:
            # 提取第一帧
            print(f"  ├─ 提取第一帧...")
            first_frame = extract_first_frame(str(video_file))

            # 检测是否包含"红松"
            print(f"  ├─ 检测红松...")
            contains_hongsong, response_text = detect_hongsong_in_image(first_frame)

            print(f"  ├─ AI回答: {response_text}")

            if contains_hongsong:
                # 重命名文件
                new_name = f"红松_{video_name}"
                new_path = video_file.parent / new_name

                # 检查新文件名是否已存在
                if new_path.exists():
                    print(f"  └─ ⚠️  跳过（目标文件已存在）: {new_name}")
                else:
                    video_file.rename(new_path)
                    print(f"  └─ ✅ 已重命名为: {new_name}")
                    renamed += 1

                detected += 1
            else:
                print(f"  └─ ❌ 未检测到红松")

            processed += 1

            # 延迟以避免API限流
            time.sleep(0.5)

        except Exception as e:
            print(f"  └─ ❌ 错误: {str(e)}")
            errors += 1
            continue

    # 输出统计
    print("\n" + "=" * 80)
    print("📊 处理完成统计")
    print("=" * 80)
    print(f"总视频数: {total}")
    print(f"已处理: {processed}")
    print(f"检测到红松: {detected}")
    print(f"已重命名: {renamed}")
    print(f"错误: {errors}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        process_videos(VIDEO_DIR)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
