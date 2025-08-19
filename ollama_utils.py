import subprocess
import shutil
import sys
import time
from typing import Optional


def ollama_cli_available() -> bool:
    """Return True if the `ollama` CLI is available on PATH."""
    return shutil.which("ollama") is not None


def get_ollama_list_output() -> str:
    """Run `ollama list` and return raw output (stdout+stderr). If CLI is missing, returns empty string."""
    if not ollama_cli_available():
        return ""
    try:
        res = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False)
        return (res.stdout or "") + (res.stderr or "")
    except Exception:
        return ""


def models_mentioned_in_list() -> list[str]:
    """Return a best-effort list of tokens (first column) from `ollama list` output."""
    out = get_ollama_list_output()
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    models = []
    for l in lines:
        parts = l.split()
        if parts:
            models.append(parts[0])
    return models


def print_restart_instructions():
    """Print user-friendly instructions for restarting Ollama on Windows (PowerShell)."""
    print("To ensure only one model is resident in GPU memory, restart your Ollama server so it doesn't keep both models loaded.")
    print("Suggested PowerShell steps:")
    print("1) Stop any running Ollama process (if present):")
    print("   Get-Process -Name ollama -ErrorAction SilentlyContinue | Stop-Process")
    print("2) Start the Ollama server (in a persistent shell):")
    print("   Start-Process -FilePath 'ollama' -ArgumentList 'serve' -NoNewWindow")
    print("Or simply reboot the machine / restart the service that runs Ollama.")


def ensure_exclusive_model(target_model: str, forbid_models: list[str] | None = None) -> None:
    """
    Ensure `target_model` is the only model appearing in `ollama list` alongside any models in `forbid_models`.
    If both target and any forbidden model are present, prints instructions and exits the process to avoid loading both into GPU.

    This function is intentionally non-destructive: it will not attempt to stop or delete models automatically.
    It only refuses to continue so the calling script doesn't accidentally cause both models to be resident.
    """
    forbid_models = forbid_models or []
    out = get_ollama_list_output().lower()

    # If we can't query the CLI, warn and continue (do not fail hard)
    if out == "":
        print("Warning: could not run `ollama list` (CLI missing or failed).")
        print("Make sure your Ollama server is running and you don't have both Teacher and Student models loaded simultaneously.")
        return

    target_low = target_model.lower()
    for forb in forbid_models:
        forb_low = forb.lower()
        if target_low in out and forb_low in out:
            print(f"Detected both models present: '{target_model}' and '{forb}' appear in 'ollama list'.")
            print("To avoid loading both models into GPU memory, stop and restart your Ollama server so only the intended model is resident.")
            print_restart_instructions()
            sys.exit(1)


### Automated convenience helpers ###
def stop_ollama() -> None:
    """Attempt to stop any running Ollama process on Windows using taskkill (best-effort)."""
    # Try to kill common process names. Use taskkill which works on Windows.
    try:
        subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], capture_output=True, text=True)
    except Exception:
        try:
            subprocess.run(["taskkill", "/F", "/IM", "ollama"], capture_output=True, text=True)
        except Exception:
            pass


def start_ollama(detach: bool = True) -> None:
    """Start `ollama serve` in a detached process so the calling script can continue.
    Uses PowerShell Start-Process on Windows for a clean background start.
    """
    if not ollama_cli_available():
        print("Warning: 'ollama' CLI not found on PATH; cannot start Ollama server automatically.")
        return
    try:
        # Use PowerShell to start process in background
        cmd = [
            "powershell",
            "-Command",
            "Start-Process -FilePath 'ollama' -ArgumentList 'serve' -NoNewWindow"
        ]
        subprocess.run(cmd, check=False)
        time.sleep(1)
    except Exception as e:
        print(f"Could not start Ollama server automatically: {e}")


def wait_for_ollama_ready(timeout: int = 30) -> bool:
    """Poll `ollama list` until we get any output or timeout. Returns True if ready."""
    start = time.time()
    while time.time() - start < timeout:
        out = get_ollama_list_output()
        if out and "models" in out.lower() or out.strip():
            return True
        time.sleep(1)
    return False


def pull_model_if_missing(model_name: str) -> None:
    """Run `ollama pull <model_name>` if the model isn't present in `ollama list` (best-effort)."""
    out = get_ollama_list_output().lower()
    if model_name.lower() in out:
        return
    if not ollama_cli_available():
        print(f"Cannot pull {model_name}: ollama CLI not found.")
        return
    try:
        print(f"Pulling model {model_name} via 'ollama pull' (this may take a while)...")
        subprocess.run(["ollama", "pull", model_name], check=False)
    except Exception as e:
        print(f"Failed to pull model {model_name}: {e}")


def ensure_only_model_loaded(target_model: str, pull_if_missing: bool = True, restart_if_needed: bool = True, forbid_models: Optional[list[str]] = None) -> None:
    """
    High-level automation: Stop Ollama, start it fresh, optionally pull the target model.
    This ensures GPU memory is freed and only the desired model will be loaded when used.

    - target_model: model to prepare (e.g., 'deepseek-r1:8b')
    - pull_if_missing: if True, run `ollama pull target_model` after restart
    - restart_if_needed: if True, always restart Ollama to free memory
    - forbid_models: deprecated; kept for compatibility
    """
    if restart_if_needed:
        print("Restarting Ollama server to ensure a clean state (will free GPU memory)...")
        stop_ollama()
        time.sleep(1)
        start_ollama()
        ready = wait_for_ollama_ready(timeout=60)
        if not ready:
            print("Warning: Ollama server did not respond after restart; continuing anyway.")

    if pull_if_missing:
        pull_model_if_missing(target_model)

