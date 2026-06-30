"""PokéMood Scanner — 启动入口
一键启动本地服务器 + Cloudflare Tunnel（公网分享）。
"""
import subprocess
import sys
import re
import os
import time
import socket
import threading

import uvicorn
import webbrowser
from threading import Timer

def get_lan_ip() -> str | None:
    """获取本机局域网 IP，失败返回 None。"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(2)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        pass

    try:
        hostname = socket.gethostname()
        candidates = socket.gethostbyname_ex(hostname)[2]
        for ip in candidates:
            if not ip.startswith("127.") and not ip.startswith("169.254."):
                return ip
    except Exception:
        pass
    return None

def _try_start_tunnel(port: int = 8000, timeout: float = 15.0) -> tuple[str | None, subprocess.Popen | None]:
    """启动 localtunnel，返回 (public_url, process)。"""
    print("[tunnel] Starting localtunnel via npx …")
    
    try:
        proc = subprocess.Popen(
            ["npx", "-y", "localtunnel", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except Exception as e:
        print(f"[tunnel] Failed to start localtunnel: {e}")
        return None, None

    url_pattern = re.compile(r"https://[a-zA-Z0-9\-]+\.loca\.lt")
    captured_url = [None]

    def reader():
        try:
            for line in proc.stdout:
                line_stripped = line.strip()
                if line_stripped:
                    print(f"  [lt] {line_stripped}")
                m = url_pattern.search(line)
                if m and captured_url[0] is None:
                    captured_url[0] = m.group(0)
        except Exception:
            pass

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    deadline = time.time() + timeout
    while captured_url[0] is None and time.time() < deadline and proc.poll() is None:
        time.sleep(0.3)

    if captured_url[0]:
        print(f"[tunnel] Public URL ready: {captured_url[0]}")
        return captured_url[0], proc
    else:
        print("[tunnel] Timed out waiting for localtunnel URL")
        return None, proc


def _print_urls(local_port: int, lan_ip: str | None, public_url: str | None):
    """在终端打印醒目的访问地址信息。"""
    lines = []
    lines.append("┌" + "─" * 55 + "┐")
    title = "  PokéMood Scanner — Access URLs"
    lines.append(f"│{title:<55}│")
    lines.append("│" + " " * 55 + "│")

    local = f"  Local:   http://localhost:{local_port}"
    lines.append(f"│{local:<55}│")

    if lan_ip:
        lan = f"  LAN:     http://{lan_ip}:{local_port}"
        lines.append(f"│{lan:<55}│")
        lines.append(f"│{'  (LAN 用户需先在浏览器添加摄像头白名单)':<55}│")

    lines.append("│" + " " * 55 + "│")
    if public_url:
        pub = f"  Public:  {public_url}"
        lines.append(f"│{pub:<55}│")
        share = "  👆 把这个链接发给任何人即可！"
        lines.append(f"│{share:<55}│")
    else:
        no_pub = "  Public:  (未启动 — cloudflared 未安装/启动失败)"
        lines.append(f"│{no_pub:<55}│")

    lines.append("└" + "─" * 55 + "┘")
    print("\n" + "\n".join(lines) + "\n")


def open_browser():
    webbrowser.open("http://localhost:8000")


if __name__ == "__main__":
    PORT = 8000

    # ─── 0. 打印启动横幅 ───
    print("\n  ⚡ PokéMood Scanner — Starting …\n")

    # ─── 1. 获取 LAN IP ───
    lan_ip = get_lan_ip()
    if lan_ip:
        print(f"[info] LAN IP: {lan_ip}")
    else:
        print("[info] Could not determine LAN IP")

    # ─── 2. 启动 cloudflared tunnel ───
    public_url, tunnel_proc = _try_start_tunnel(PORT, timeout=15.0)

    # ─── 3. 打印访问地址 ───
    _print_urls(PORT, lan_ip, public_url)

    # ─── 4. 自动打开浏览器 ───
    Timer(1.5, open_browser).start()

    # ─── 5. 启动 uvicorn ───
    print("[server] Starting uvicorn on 0.0.0.0:8000 …\n")
    try:
        uvicorn.run("backend.server:app", host="0.0.0.0", port=PORT, reload=False)
    finally:
        # 退出时清理 tunnel 进程
        if tunnel_proc and tunnel_proc.poll() is None:
            print("[tunnel] Shutting down …")
            tunnel_proc.terminate()
            tunnel_proc.wait(timeout=5)
