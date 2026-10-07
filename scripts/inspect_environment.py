"""Read-only environment inventory. Does not allocate model weights."""
import importlib
import json
import sys

report = {"python": sys.executable, "version": sys.version}
for name in ("torch", "transformers", "peft", "pytest", "av", "soundfile", "ltx_core"):
    try:
        mod = importlib.import_module(name)
        report[name] = {"version": getattr(mod, "__version__", None), "file": mod.__file__}
    except Exception as exc:
        report[name] = {"error": str(exc)}
print(json.dumps(report, indent=2))
