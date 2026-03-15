import pickle
import sys
import os
import http.cookies

def create_cookies_file():
    # 尝试从同级目录或根目录读取 cookies.txt 或 cookie.txt
    cookie_string = ""
    for filename in ["cookies.txt", "cookie.txt", "tools/cookie.txt"]:
        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    cookie_string = f.read().strip()
                print(f"✅ 已成功从 {filename} 读取 Cookie 数据。")
                break
            except Exception as e:
                print(f"尝试读取 {filename} 时出错: {e}")
    
    if not cookie_string:
        print("未找到本地的 cookies.txt，请手动输入你在网站上获取的完整 Cookie 字符串:")
        cookie_string = input("> ").strip()
    
    if not cookie_string:
        print("未获取到任何内容，程序退出。")
        return

    # 将字符串解析成带域名的 SimpleCookie 格式
    cookie_dict = http.cookies.SimpleCookie()
    try:
        parts = cookie_string.split(';')
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                key = key.strip()
                cookie_dict[key] = value.strip()
                # 【关键修复】手动绑定域名
                cookie_dict[key]['domain'] = "shuiyuan.sjtu.edu.cn"
                cookie_dict[key]['path'] = "/"
    except Exception as e:
        print(f"解析 Cookie 字符串失败: {e}")
        return

    print("\n解析出的 Cookie 字段如下：")
    for k, v in cookie_dict.items():
        val = v.value
        print(f"[{k}]: {val[:20]}{'...' if len(val)>20 else ''}")

    # 以二进制写入到项目的 cookies 文件中使用 pickle 序列化
    try:
        with open("cookies", "wb") as f:
            pickle.dump(cookie_dict, f)
        print("\n✅ 成功创建并写入 'cookies' 文件，程序现在能够正确读取它了！")
    except Exception as e:
        print(f"\n❌ 写入文件失败: {e}")

if __name__ == "__main__":
    create_cookies_file()