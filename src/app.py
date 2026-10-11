# Forwarding to main root app.py to ensure identical behavior
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_APP = os.path.join(BASE_DIR, "app.py")

with open(ROOT_APP, "r", encoding="utf-8") as f:
    code = f.read()

exec(compile(code, ROOT_APP, "exec"))
