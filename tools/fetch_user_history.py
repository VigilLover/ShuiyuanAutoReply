import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.shuiyuan.shuiyuan_model import ShuiyuanModel
from dataclasses import asdict

import aiohttp
import asyncio
from datetime import datetime, timezone
import json
import re
import csv
import pickle

def parse_iso_datetime(dt_str: str) -> datetime | None:
    try:
        return datetime.strptime(dt_str.replace('Z', '+0000'), '%Y-%m-%dT%H:%M:%S.%f%z')
    except Exception:
        try:
            return datetime.strptime(dt_str.replace('Z', '+0000'), '%Y-%m-%dT%H:%M:%S%z')
        except Exception:
            return None

async def get_user_replies(username: str, max_pages: int | None = None,
                          since_dt: datetime | None = None,
                          until_dt: datetime | None = None) -> list[dict]:
    print(f'正在异步获取用户 @{username} 的回复...')
    
    all_replies = []
    offset = 0
    page = 1
    
    # 使用 shuiyuan_model 进行全局实例化并自行处理持久化 Cookie / CSRF
    model = await ShuiyuanModel.create("cookies")
    
    # 动态调低 ShuiyuanModel 类的全局请求间隔以加快爬取速度（默认是保守的 1.0 秒以防封禁）
    # 但需注意不能调得太低，0.3秒是一个相对适中的速度
    ShuiyuanModel._request_interval = 0.3
    
    while True:
        if max_pages and page > max_pages:
            break
            
        try:
            max_retries = 5
            user_actions = None
            
            for attempt in range(max_retries):
                try:
                    # filter=5 for replies
                    actions_obj = await model.get_actions(username=username, filter=[5], offset=offset)
                    user_actions = [asdict(ua) for ua in actions_obj.user_actions]
                    break
                except Exception as e:
                    print(f"遇到错误: {e}，等待 2 秒后重试... ({attempt + 1}/{max_retries})")
                    await asyncio.sleep(2)
            
            if user_actions is None:
                print(f"最终请求失败，跳出当前爬取循环。")
                break
                
            if len(user_actions) == 0:
                print(f"已获取所有回复，共 {len(all_replies)} 条")
                break
                
            filtered_actions = []
            page_times = []
            for ua in user_actions:
                c = ua.get('created_at')
                cdt = parse_iso_datetime(c) if c else None
                if cdt:
                    page_times.append(cdt)
                    if since_dt and cdt < since_dt:
                        continue
                    if until_dt and cdt > until_dt:
                        continue
                filtered_actions.append(ua)

            all_replies.extend(filtered_actions)
            print(f"第 {page} 页: 获取了 {len(user_actions)} 条，窗口内 {len(filtered_actions)} 条 (累计 {len(all_replies)} 条)")

            if since_dt and page_times:
                oldest_on_page = min(page_times)
                if oldest_on_page < since_dt:
                    print("达到开始时间阈值，停止翻页。")
                    break
            
            offset += 30
            page += 1
                
        except Exception as e:
            print(f"获取第 {page} 页时出错: {e}")
            break
    
    return all_replies

def process_replies(username: str, data: list[dict]):
    posts_text = []
    posts_csv = []
    
    aPattern = re.compile(r'<a.*?>.*?</a>')
    emojiPattern = re.compile(r'<img[^>]*title="(:[^:]*:)"[^>]*>')
    htmlPattern = re.compile(r'<[^>\n]*>')
    # 匹配完整的 div data-signature 块，不区分大小写，包含换行符
    signaturePattern = re.compile(r'<div\s+data-signature[^>]*>.*?</div>', re.DOTALL | re.IGNORECASE)
    
    for item in data:
        content = item.get('excerpt', '')
        # 如果爬取到的是 raw 的话会有完整的 div 结构，直接正则替换为空
        content = signaturePattern.sub('', content)
        
        # 移除可能被 API 剥离了 div 标签但保留了内部文本的残余签名
        content = re.sub(r'\[right\].*?\[/right\]', '', content, flags=re.IGNORECASE)
        content = content.replace('这里是中杯小狼(>^ω^<)', '')
        content = content.replace('这里是中杯小狼(&gt;^ω^&lt;)', '')
        
        # 移除总结
        content = content.replace('▶ \n总结\n', '').replace('▶\n总结\n', '').replace('▶ \n总结', '').replace('▶\n总结', '')
        
        content = content.replace('&hellip;', '')
        content = aPattern.sub('', content)
        while emojiPattern.search(content):
            content = emojiPattern.sub(lambda m: m.group(1), content, 1)
        
        # 将不可见字符或html实体替换处理即可
        # 兼容示范文本中的换行保留逻辑
        
        content = content.strip()
        if content and content != '':
            posts_text.append(content + '\n')
            posts_csv.append([content])

    archive_dir = os.path.join("user_archive", username)
    os.makedirs(archive_dir, exist_ok=True)

    # 写入txt文件
    with open(os.path.join(archive_dir, f'{username}_posts.txt'), 'w', encoding='utf-8') as f:
        for post in posts_text:
            f.write(post + '\n')
            
    # 写入csv文件
    with open(os.path.join(archive_dir, f'{username}_posts.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['post_raw'])
        writer.writerows(posts_csv)
        
    print(f"文本已保存到 {archive_dir} 下的 {username}_posts.txt 和 {username}_posts.csv")

if __name__ == "__main__":
    username = input("请输入水源用户名: ").strip()
    since_input = input("请输入开始时间 (YYYY-MM-DD) 或留空: ").strip()
    until_input = input("请输入结束时间 (YYYY-MM-DD) 或留空: ").strip()
    
    since_dt = datetime.strptime(since_input, '%Y-%m-%d') if since_input else None
    until_dt = datetime.strptime(until_input, '%Y-%m-%d') if until_input else None
    
    if since_dt and since_dt.tzinfo is None:
        since_dt = since_dt.replace(tzinfo=timezone.utc)
    if until_dt and until_dt.tzinfo is None:
        until_dt = until_dt.replace(tzinfo=timezone.utc)
    
    replies = asyncio.run(get_user_replies(username, since_dt=since_dt, until_dt=until_dt))
    
    archive_dir = os.path.join("user_archive", username)
    os.makedirs(archive_dir, exist_ok=True)
    
    # 也可以保存为原始json便于查看
    with open(os.path.join(archive_dir, f"{username}_replies.json"), "w", encoding="utf-8") as f:
        json.dump(replies, f, ensure_ascii=False, indent=4)
        
    process_replies(username, replies)
