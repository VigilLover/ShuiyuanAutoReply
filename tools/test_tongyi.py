import os
import asyncio
import dotenv
from openai import AsyncOpenAI

# 从项目根目录或被执行目录的 .env 文件加载环境变量
dotenv.load_dotenv()

async def main():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误：环境变量中未找到 DASHSCOPE_API_KEY，请确保已设置。")
        return
        
    print("正在初始化 AsyncOpenAI 客户端...")
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    print("正在发送请求测试连通性...")
    try:
        response = await client.chat.completions.create(
            model="qwen-plus-2025-07-28", # 也可以用项目中出现的 qwen-plus-2025-07-28 或 deepseek-v3.1
            messages=[
                {"role": "user", "content": "你好，请回复“API调用成功！”用于测试模型连通性。并介绍一下上海交通大学。"}
            ]
        )
        print("\n=== 调用成功 ===")
        print("大模型回复内容：")
        print(response.choices[0].message.content)
        print("=================\n")
    except Exception as e:
        print("\n=== 调用失败 ===")
        print(f"发生错误: {e}")
        print("=================\n")

if __name__ == "__main__":
    asyncio.run(main())
