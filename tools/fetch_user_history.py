import aiohttp
import asyncio
from datetime import datetime, timezone
import json
import re
import csv
import os

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
    
    try:
        with open("cookie.txt", "r", encoding="utf-8") as f:
            cookie = f.read().strip()
    except FileNotFoundError:
        print("错误: 未找到 cookie.txt 文件")
        return []
    
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36",
        "cookie": cookie
    }
    
    async with aiohttp.ClientSession(headers=headers) as session:
        while True:
            if max_pages and page > max_pages:
                break
                
            url = f"https://shuiyuan.sjtu.edu.cn/user_actions.json?username={username}&filter=5&offset={offset}"
            
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        print(f"请求失败: {response.status if response else 'Network Error'}")
                        break
                        
                    data = await response.json()
                    user_actions = data.get('user_actions', [])
                    
                    if not user_actions:
                        print(f"已获取所有回复，共 {len(all_replies)} 条")
                        break
                    
                    filtered_actions = []
                    page_times = []
                    for ua in user_actions:
                        c = ua.get('created_at')
                        cdt = parse_iso_datetime(c) if c else None
                        if cdt:
                            page_times.append(cdt)
                        if cdt:
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
                    
                    await asyncio.sleep(0.3)
                    
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

    # 写入txt文件
    with open(f'user_archive.txt', 'w', encoding='utf-8') as f:
        for post in posts_text:
            f.write(post + '\n')
            
    # 写入csv文件
    with open(f'user_archive.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['post_raw'])
        writer.writerows(posts_csv)
        
    print(f"文本已保存到 user_archive.txt 和 user_archive.csv")

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
    
    # 也可以保存为原始json便于查看
    with open(f"user_archive.json", "w", encoding="utf-8") as f:
        json.dump(replies, f, ensure_ascii=False, indent=4)
        
    process_replies(username, replies)
