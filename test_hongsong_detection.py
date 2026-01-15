#!/usr/bin/env python3
"""
测试红松检测 - 只处理前5个视频
"""
import cv2
import requests
import base64
import json
from pathlib import Path
from typing import Tuple

# 配置
API_BASE = "http://localhost:8000"
API_KEY = "sk-demo-token"
MODEL = "gemini-2.5-flash"
VIDEO_DIR = "/Users/chen/ClaudeCode/工作项目/行业素材库/乐器行业/视频数据2"
TEST_LIMIT = 5

# 检测提示词
DETECTION_PROMPT = """请仔细查看这张图片，检测是否包含"红松"这两个字或者红松品牌的logo。

要求：
1. 如果图片中包含"红松"文字（简体或繁体），回答：是
2. 如果图片中包含红松品牌的logo或标识，回答：是
3. 如果都不包含，回答：否

只需要回答"是"或"否"，不要有其他内容。"""


def extract_first_frame(video_path: str) -> bytes:
    """提取视频第一帧并返回JPG格式的字节数据"""
    print(f"    ├─ 打开视频...")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception(f"无法打开视频")

    # 读取第一帧
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise Exception(f"无法读取视频帧")

    print(f"    ├─ 帧尺寸: {frame.shape[1]}x{frame.shape[0]}")

    # 转为JPG格式
    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        raise Exception(f"无法编码图片")

    return buffer.tobytes()


def detect_hongsong_in_image(image_data: bytes) -> Tuple[bool, str]:
    """
    使用Gemini API检测图片中是否包含"红松"
    """
    try:
        # 转为base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        print(f"    ├─ Base64长度: {len(image_base64):,}")

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
                "temperature": 0.1,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 100,
            }
        }

        print(f"    ├─ 调用Gemini API...")
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


def test_detection():
    """测试检测功能"""
    print("=" * 80)
    print("🎬 红松视频检测 - 测试模式")
    print("=" * 80)

    # 获取视频文件
    video_dir_path = Path(VIDEO_DIR)
    video_files = sorted(video_dir_path.glob("*.mp4"))[:TEST_LIMIT]

    print(f"\n📂 目录: {VIDEO_DIR}")
    print(f"🔍 测试数量: {len(video_files)}")
    print(f"🤖 使用模型: {MODEL}")
    print("\n" + "=" * 80)

    # 处理每个视频
    for idx, video_file in enumerate(video_files, 1):
        video_name = video_file.name

        print(f"\n[{idx}/{len(video_files)}] 📹 {video_name}")

        try:
            # 提取第一帧
            first_frame = extract_first_frame(str(video_file))

            # 检测
            contains_hongsong, response_text = detect_hongsong_in_image(first_frame)

            print(f"    ├─ AI回答: {response_text}")

            if contains_hongsong:
                print(f"    └─ ✅ 检测到红松！")
            else:
                print(f"    └─ ❌ 未检测到红松")

        except Exception as e:
            print(f"    └─ ❌ 错误: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("✅ 测试完成")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_detection()
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
