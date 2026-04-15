#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-rocm"
RUNTIME_DIR="${ROOT_DIR}/.rocm-wsl/extracted/opt/rocm-6.4.2/lib"

echo "== System =="
date '+%Y-%m-%d %H:%M:%S %Z (%z)'
python3 --version || true
python3.12 --version || true
grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release || true

echo
echo "== Windows GPU =="
powershell.exe -NoProfile -Command "Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion | Format-Table -AutoSize" || true

echo
echo "== Current WSL device nodes =="
for p in /dev/dxg /dev/kfd /dev/dri; do
  if [[ -e "${p}" ]]; then
    echo "${p}: present"
  else
    echo "${p}: missing"
  fi
done

echo
echo "== Official AMD Notes =="
echo "RX 9070 XT is listed in AMD's WSL support matrix for ROCm 7.2.1."
echo "PyTorch 2.9.1 + ROCm 7.2.1 is the official production support combo."
echo "Ubuntu 24.04 wheels use CPython 3.12."

echo
echo "== Local ROCm Python Env =="
if [[ -x "${VENV_DIR}/bin/python" ]]; then
  "${VENV_DIR}/bin/python" --version
else
  echo "missing: ${VENV_DIR}"
fi

echo
echo "== Local WSL Runtime Extraction =="
if [[ -d "${RUNTIME_DIR}" ]]; then
  find "${RUNTIME_DIR}" -maxdepth 1 -type f | sed -n '1,20p'
else
  echo "missing: ${RUNTIME_DIR}"
fi

cat <<'EOF'

Suggested next steps:

1. If you can use sudo interactively, install AMD's official WSL runtime:
   wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb
   sudo apt install ./amdgpu-install_7.2.1.70201-1_all.deb
   sudo amdgpu-install -y --usecase=wsl,rocm --accept-eula

2. Re-open the WSL shell and verify:
   . .venv-rocm/bin/activate
   python -c "import torch; print(torch.cuda.is_available(), getattr(torch.version, 'hip', None), torch.cuda.get_device_name(0))"

3. If sudo is unavailable, you can still inspect missing runtime libs with:
   ldd .venv-rocm/lib/python3.12/site-packages/torch/_C.cpython-312-x86_64-linux-gnu.so

This project itself uses the generic torch.cuda path, so once ROCm torch imports cleanly the trainer will use the AMD GPU automatically.
EOF
