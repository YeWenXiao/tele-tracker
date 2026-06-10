#!/usr/bin/env python3
"""phone_upload_server.py — 手机局域网传图到 Jetson。
手机浏览器打开 http://<Jetson IP>:8888 → 选照片 → 上传到 uploads/日期/。
纯标准库,零依赖。Ctrl+C 退出。
"""
import os, re, time, socket, subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = 8888
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')

PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>传图到 Jetson</title>
<style>
body{font-family:sans-serif;max-width:480px;margin:20px auto;padding:0 16px;background:#111;color:#eee}
h2{color:#7c4}
.box{border:2px dashed #555;border-radius:12px;padding:24px;text-align:center;margin:16px 0}
input[type=file]{width:100%%;padding:12px 0}
button{width:100%%;padding:16px;font-size:20px;background:#7c4;color:#111;border:0;border-radius:10px;font-weight:bold}
.ok{background:#163;border-radius:8px;padding:10px;margin:8px 0}
.note{color:#888;font-size:14px}
</style></head>
<body>
<h2>📷 传图到 Jetson</h2>
%s
<form method="POST" action="/upload" enctype="multipart/form-data">
<div class="box">
<input type="file" name="photos" accept="image/*,video/*" multiple>
</div>
<button type="submit">上传</button>
</form>
<p class="note">支持多选照片/视频 · 存到 uploads/%s/</p>
</body></html>"""


def today():
    return time.strftime('%Y%m%d')


class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        pass

    def _html(self, body):
        data = body.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._html(PAGE % ('', today()))

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        ctype = self.headers.get('Content-Type', '')
        m = re.search(r'boundary=([^;]+)', ctype)
        if not m or length == 0:
            self._html(PAGE % ('<div class="ok">没有收到文件</div>', today()))
            return
        boundary = m.group(1).strip('"').encode()
        body = self.rfile.read(length)
        day_dir = os.path.join(BASE, today())
        os.makedirs(day_dir, exist_ok=True)
        saved = []
        # 手工解 multipart(零依赖):按 boundary 切段,找 filename
        for part in body.split(b'--' + boundary):
            if b'filename="' not in part:
                continue
            head, _, payload = part.partition(b'\r\n\r\n')
            fn = re.search(rb'filename="([^"]*)"', head)
            if not fn or not fn.group(1):
                continue
            name = os.path.basename(fn.group(1).decode('utf-8', 'ignore')) or f'img_{int(time.time())}'
            payload = payload.rstrip(b'\r\n')
            if not payload:
                continue
            # 重名加时间戳
            path = os.path.join(day_dir, name)
            if os.path.exists(path):
                stem, ext = os.path.splitext(name)
                path = os.path.join(day_dir, f"{stem}_{time.strftime('%H%M%S')}{ext}")
            with open(path, 'wb') as f:
                f.write(payload)
            saved.append((os.path.basename(path), len(payload)))
            print(f"[收到] {path} ({len(payload)/1024:.0f} KB)")
        if saved:
            items = ''.join(f'<div class="ok">✅ {n} ({s//1024} KB)</div>' for n, s in saved)
            msg = f'{items}<div class="note">共 {len(saved)} 个,继续传或关掉页面</div>'
        else:
            msg = '<div class="ok">没有收到文件</div>'
        self._html(PAGE % (msg, today()))


def all_ips():
    try:
        out = subprocess.check_output(['hostname', '-I'], text=True).split()
        return [ip for ip in out if not ip.startswith('127.')]
    except Exception:
        return [socket.gethostbyname(socket.gethostname())]


if __name__ == '__main__':
    os.makedirs(BASE, exist_ok=True)
    print('=' * 46)
    print('  手机传图服务已启动(手机和 Jetson 连同一 WiFi)')
    for ip in all_ips():
        print(f'  手机浏览器打开:  http://{ip}:{PORT}')
    print(f'  照片存到: {BASE}/{today()}/')
    print('  Ctrl+C 退出')
    print('=' * 46)
    HTTPServer(('0.0.0.0', PORT), H).serve_forever()
