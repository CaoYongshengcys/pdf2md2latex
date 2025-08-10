print("检查Python环境...")

try:
    import os
    print("os模块: OK")
except ImportError as e:
    print(f"os模块导入失败: {e}")

try:
    import subprocess
    print("subprocess模块: OK")
except ImportError as e:
    print(f"subprocess模块导入失败: {e}")

try:
    import re
    print("re模块: OK")
except ImportError as e:
    print(f"re模块导入失败: {e}")

try:
    import PyPDF2
    print("PyPDF2模块: OK")
except ImportError as e:
    print(f"PyPDF2模块导入失败: {e}")

try:
    import pdfplumber
    print("pdfplumber模块: OK")
except ImportError as e:
    print(f"pdfplumber模块导入失败: {e}")

try:
    from flask import Flask
    print("Flask模块: OK")
except ImportError as e:
    print(f"Flask模块导入失败: {e}")

print("检查完成")