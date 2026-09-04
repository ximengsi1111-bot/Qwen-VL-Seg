from __future__ import annotations

import platform
import sys


def main() -> None:
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")
    try:
        import torch

        print(f"torch={torch.__version__}")
        print(f"cuda_available={torch.cuda.is_available()}")
        print(f"cuda_version={torch.version.cuda}")
        print(f"device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    except Exception as exc:  # pragma: no cover - diagnostics
        print(f"torch_import_error={exc}")
    for name in ("transformers", "peft", "accelerate", "deepspeed"):
        try:
            module = __import__(name)
            print(f"{name}={getattr(module, '__version__', 'unknown')}")
        except Exception as exc:
            print(f"{name}_import_error={exc}")


if __name__ == "__main__":
    main()

