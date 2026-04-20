"""Patch uvicorn config to add LoopSetupType alias for acp-sdk compatibility."""
import site, os

# Find the uvicorn config.py in user site-packages
candidates = [
    os.path.join(site.getusersitepackages(), "uvicorn", "config.py"),
    os.path.join(site.getsitepackages()[0], "uvicorn", "config.py"),
]

patched = False
for path in candidates:
    if not os.path.exists(path):
        continue
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if "LoopSetupType" in content:
        print(f"Already patched: {path}")
        patched = True
        break
    old = 'LoopFactoryType = Literal["none", "auto", "asyncio", "uvloop"]'
    new = old + "\nLoopSetupType = LoopFactoryType  # compat alias for acp-sdk"
    if old in content:
        content = content.replace(old, new)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Patched: {path}")
        patched = True
        break

if not patched:
    print("Could not find uvicorn/config.py to patch.")
