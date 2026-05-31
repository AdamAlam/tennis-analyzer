"""Verify the PyTorch + CUDA install actually drives the RTX 5090 (Blackwell sm_120).

Run AFTER installing torch from the cu128 index:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    python scripts/check_cuda.py

Expected on a 5090: cuda available: True, device 'NVIDIA GeForce RTX 5090', capability (12, 0).
"""

from __future__ import annotations


def main() -> None:
    import torch

    print(f"torch:          {torch.__version__}")
    print(f"cuda runtime:   {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print(
            "\nCUDA is NOT available. On a 5090 this almost always means the wrong torch wheel.\n"
            "Reinstall from the cu128 index:\n"
            "  pip uninstall -y torch torchvision\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
        )
        return

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"device:         {name}")
    print(f"capability:     {cap}  (expect (12, 0) on Blackwell / RTX 5090)")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")

    # Real kernel launch: if the wheel lacks sm_120 kernels this raises 'no kernel image'.
    try:
        x = torch.randn(4096, 4096, device="cuda")
        val = (x @ x).sum().item()
        print(f"matmul on cuda: OK ({val:.1f})")
        print("\nAll good - the GPU is being used.")
    except RuntimeError as exc:
        print(f"\nmatmul FAILED on cuda: {exc}")
        print("This is the classic sm_120 mismatch - install the cu128 torch build.")


if __name__ == "__main__":
    main()
