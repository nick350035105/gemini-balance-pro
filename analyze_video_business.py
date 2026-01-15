#!/usr/bin/env python3
"""
使用Gemini Balance服务进行业务场景化的视频分析
"""
import requests
import json
import base64
import mimetypes
from pathlib import Path

# 配置
API_BASE = "http://localhost:8000"
API_KEY = "sk-demo-token"
MODEL = "gemini-flash-latest"
VIDEO_PATH = "/Users/chen/Downloads/530917039355789390.mp4"

# 业务场景化的提示词
BUSINESS_PROMPT = """##目标：逐帧分析视频，并将其拆解成一份结构清晰、信息完整、符合专业制作标准的分镜脚本
###分镜脚本必须精准、结构化，指定的Markdown表格格式。
| boardNo | startTime | endTime | speaker | subtitle | scene | isFloat | straightOn |
|---|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... | ... |
##字段定义:
1-speaker (讲述人): 识别语音来源,注意区分说话角色,画外音固定使用 "旁白"。
2-subtitle (字幕): 识别配音的字幕,镜头内出现的完整口播文字。请使用 \\ 来模拟视觉上的字幕换行，以增强可读性。
--前贴和后贴的subtitle，如果没有配音字幕则返回空，不要返回画面的文字
3-scene (画面描述): 简明扼要地描述当前镜头的视觉内容。包括：环境、人物、关键动作、核心物体、以及任何重要的图形或文字信息。
4-isFloat (是否浮屏- 布尔值)：true/false,和画面描述的浮屏/浮窗关联，当画面是浮屏/浮窗时返回true
5-straightOn (是否正打 - 布尔值): 
--straightOn 字段判断的黄金法则:
标记为 true 的情况:画面中的主体人物正在进行标准的口播讲述，人物是画面的焦点。侧打的情况也归类为正打true
标记为 false 的所有情况:语音是画外音、画面中没有人，或只有背景人物、人物在画面中，但没有说话。
6-【浮屏描述】 如果主讲人是以**“浮窗”、“画中画”或“小窗”**的形式叠加在主画面上进行口播，即使他/她正视镜头，straightOn 也必须标记为 false。这种形式不被定义为“正打”。
7-前后贴识别规则:
--前贴 (Pre-roll): 指视频开头的、与核心内容分离的独立片头，如品牌Logo演绎、活动标题、吸睛片段、高能片段、名人采访等。如果识别出前贴，必须在该分镜的 scene 描述最前方加上 【前贴】 标签。
--后贴 (Post-roll / End-card): 指视频结尾用于品牌强化和引导转化的静态或动态画面，通常包含Logo、Slogan和明确的行动号召（如“立即下载”）。如果识别出后贴，必须在该分镜的 scene 描述最前方加上 【后贴】 标签。
****注意：一个视频最多只有1个前贴分镜或1个后贴分镜，禁止拆前贴和后贴的分镜，前贴存在转场也不允许拆前贴分镜，完整的前贴作为第一个分镜，完整的后贴作为最后一个分镜****
####strategyType: 素材类型，1=空镜混剪，2=数字人口播，3=实拍真人口播，0=非口播视频（除了前贴和后贴，其他分镜字幕提取不到则为非口播视频）
8-素材类型判：
--空镜混剪：非数字人口播且非真人口播，没有人物正打镜头
--数字人口播：存在数字人正打说话镜头，素材标题带有数字人，统一识别为数字人
--实拍真人口播：存在真人正打说话镜头（注意严格区分时数字人还是真人，不要把数字人识别为真人）
（区分真人和数字人的方式：**眼神、皮肤、面部细节与微表情、人物背景）

##分镜脚本工作流程:
1-顺序分析: 从视频的第0秒开始，按时间顺序处理。
****识别切点: 精准定位每一个镜头画面切换的时刻，以此划分不同的 boardNo。****
2-特殊切点：当分镜时长超过5秒没有镜头切换，则在下一个字幕切换处划分不同的boardNo。
3-处理分镜: 对每一个独立的镜头，执行以下操作：
a. 记录 startTime 和 endTime。
b. 提取 subtitle 并识别 speaker。
c. 分析视觉元素，撰写 scene 描述。
d. 根据画面是否浮屏判断 isFloat 。
e. 应用 straightOn 黄金法则: 这是最关键的一步。仔细评估人物是否为画面主体，以及他/她的视线方向。特别要检查是否存在“浮窗”口播的特殊情况。
f. 补充场景描述: 当 straightOn 因“浮窗”规则被设为 false 时，必须在 scene 字段中用文字补充说明这一视觉形式，例如：“画面右下角有主播浮窗讲解”或“数字人浮屏介绍产品特性”。

# 强制输出JSON格式，其他禁止任何解释和说明：
 
      {
        "boardNo": <分镜编号>,
        "startTime": <startTime，格式为 HH:MM:SS.ms>,
        "endTime": <endTime，格式为 HH:MM:SS.ms>,
        "speaker":  <说话人>,
         "subtitle": <分镜字幕>,
         "scene": <分镜画面描述>,
         "isFloat": <是否浮屏>,
         "straightOn": <是否正打镜头>,
         "modifyType": <分镜脚本修改建议，0代表不修改/1代表修改空镜/2代表修改数字人/-1代表前贴或者后贴分镜>
      }" """


def analyze_video_with_business_context():
    """使用业务场景化提示词分析视频"""
    print("=" * 70)
    print("🎬 Gemini Balance - 业务场景化视频分析")
    print("=" * 70)

    video_file = Path(VIDEO_PATH)
    if not video_file.exists():
        print(f"❌ 视频文件不存在: {VIDEO_PATH}")
        return

    print(f"\n📹 视频文件: {video_file.name}")
    print(f"   文件大小: {video_file.stat().st_size / 1024 / 1024:.2f} MB")

    # 读取视频文件并转为base64
    print(f"\n📤 正在读取视频文件...")
    with open(VIDEO_PATH, 'rb') as f:
        video_data = f.read()

    video_base64 = base64.b64encode(video_data).decode('utf-8')
    mime_type = mimetypes.guess_type(VIDEO_PATH)[0] or "video/mp4"

    print(f"   MIME类型: {mime_type}")
    print(f"   Base64编码长度: {len(video_base64):,} 字符")

    print(f"\n💡 使用提示词:")
    print("-" * 70)
    print(BUSINESS_PROMPT)
    print("-" * 70)

    print(f"\n🤖 调用 Gemini API ({MODEL})...")

    # 使用Gemini原生API格式
    request_body = {
        "contents": [{
            "parts": [
                {"text": BUSINESS_PROMPT},
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
            "maxOutputTokens": 204800,
        }
    }

    response = requests.post(
        f"{API_BASE}/gemini/v1beta/models/{MODEL}:generateContent",
        headers={
            "x-goog-api-key": API_KEY,
            "Content-Type": "application/json"
        },
        json=request_body,
        timeout=180
    )

    print(f"\n📊 响应状态码: {response.status_code}")

    if response.status_code != 200:
        print(f"❌ 请求失败")
        print(f"响应: {response.text}")
        return

    result = response.json()

    # 先打印完整响应用于调试
    print(f"\n🔍 完整响应:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 提取文本内容
    try:
        text_content = result["candidates"][0]["content"]["parts"][0]["text"]

        print(f"\n" + "=" * 70)
        print("📝 业务场景化分析结果")
        print("=" * 70)
        print(text_content)
        print("=" * 70)

        # 显示token使用情况
        if "usageMetadata" in result:
            usage = result["usageMetadata"]
            print(f"\n📊 Token使用统计:")
            print(f"   提示词Token: {usage.get('promptTokenCount', 0):,}")
            print(f"   生成Token: {usage.get('candidatesTokenCount', 0):,}")
            print(f"   总计Token: {usage.get('totalTokenCount', 0):,}")

        print(f"\n✅ 分析完成!")

    except (KeyError, IndexError) as e:
        print(f"❌ 解析响应失败: {e}")
        print(f"\n完整响应:")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        analyze_video_with_business_context()
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
