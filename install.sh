#!/usr/bin/env bash
# gpusched installer
# curl -sSL https://raw.githubusercontent.com/shayonj/gpusched/main/install.sh | sudo bash
#
# Installs:
#   1. gpusched binary → /usr/local/bin/gpusched
#   2. cuda-checkpoint → /usr/local/bin/cuda-checkpoint
#   3. CRIU (via apt/dnf)
#   4. systemd service → /etc/systemd/system/gpusched.service
#   5. Working directories → /var/lib/gpusched/
set -euo pipefail

GPUSCHED_VERSION="${GPUSCHED_VERSION:-latest}"
GITHUB_REPO="shayonj/gpusched"
INSTALL_DIR="/usr/local/bin"
DATA_DIR="/var/lib/gpusched"
LOG_DIR="/var/log/gpusched"

# ── Colors ───────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${GREEN}▸${NC} $*"; }
warn()  { echo -e "${YELLOW}▸${NC} $*"; }
error() { echo -e "${RED}✗${NC} $*" >&2; exit 1; }
step()  { echo -e "\n${CYAN}${BOLD}[$1/$TOTAL] $2${NC}"; }

# ── Preflight ────────────────────────────────────────────────────────────────

echo -e "${BOLD}"
echo '  ╔═╗╔═╗╦ ╦╔═╗╔═╗╦ ╦╔═╗╔╦╗'
echo '  ║ ╦╠═╝║ ║╚═╗║  ╠═╣║╣  ║║'
echo '  ╚═╝╩  ╚═╝╚═╝╚═╝╩ ╩╚═╝═╩╝'
echo -e "${NC}"
echo -e "  GPU Process Manager"
echo ""

if [ "$(id -u)" -ne 0 ]; then
    error "This installer must be run as root. Try: curl ... | sudo bash"
fi

if [ "$(uname -s)" != "Linux" ]; then
    error "gpusched requires Linux. Got: $(uname -s)"
fi

if ! command -v curl &>/dev/null; then
    error "curl is required. Install it with: apt install curl"
fi

if ! command -v tar &>/dev/null; then
    error "tar is required. Install it with: apt install tar"
fi

TOTAL=7

# ── Step 1: Check GPU ────────────────────────────────────────────────────────

step 1 "Checking GPU and driver"

if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi not found. Install NVIDIA drivers first."
fi

DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d. -f1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

if [ "$DRIVER_MAJOR" -lt 580 ]; then
    error "Driver $DRIVER_VERSION too old. gpusched requires 580+ for cuda-checkpoint."
fi

info "Driver: $DRIVER_VERSION"
info "GPU:    $GPU_NAME ($GPU_MEM MB)"

# ── Step 2: Install cuda-checkpoint ──────────────────────────────────────────

step 2 "Installing cuda-checkpoint"

if command -v cuda-checkpoint &>/dev/null; then
    info "Already installed: $(which cuda-checkpoint)"
else
    CKPT_ARCH="$(uname -m)_$(uname -s)"
    CKPT_URL="https://raw.githubusercontent.com/NVIDIA/cuda-checkpoint/main/bin/${CKPT_ARCH}/cuda-checkpoint"
    if curl -fsSL "$CKPT_URL" -o "$INSTALL_DIR/cuda-checkpoint"; then
        chmod 755 "$INSTALL_DIR/cuda-checkpoint"
        info "Installed: $INSTALL_DIR/cuda-checkpoint"
    else
        warn "No prebuilt cuda-checkpoint for $CKPT_ARCH"
        warn "See: https://github.com/NVIDIA/cuda-checkpoint"
    fi
fi

# ── Step 3: Install CRIU ─────────────────────────────────────────────────────

step 3 "Installing CRIU"

if command -v criu &>/dev/null; then
    info "Already installed: $(criu --version 2>&1 | head -1)"
else
    if command -v apt-get &>/dev/null; then
        apt-get update -qq 2>/dev/null && apt-get install -y -qq criu 2>/dev/null
    elif command -v dnf &>/dev/null; then
        dnf install -y -q criu 2>/dev/null
    elif command -v yum &>/dev/null; then
        yum install -y -q criu 2>/dev/null
    else
        warn "Cannot auto-install CRIU. Install manually for fork/hibernate."
    fi
    if command -v criu &>/dev/null; then
        info "Installed: $(criu --version 2>&1 | head -1)"
    fi
fi

# ── Step 4: Resolve version ──────────────────────────────────────────────────

step 4 "Resolving gpusched version"

if [ "$GPUSCHED_VERSION" = "latest" ]; then
    GPUSCHED_VERSION=$(curl -fsSL "https://api.github.com/repos/${GITHUB_REPO}/releases/latest" \
        | grep '"tag_name"' | head -1 | sed 's/.*"v\([^"]*\)".*/\1/' || echo "")
    if [ -z "$GPUSCHED_VERSION" ]; then
        error "Could not determine latest version. Set GPUSCHED_VERSION=x.y.z and retry."
    fi
fi

info "Version: v${GPUSCHED_VERSION}"

# ── Step 5: Install gpusched binary ──────────────────────────────────────────

step 5 "Installing gpusched"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo /tmp)"
ARCH="$(uname -m)"
case "$ARCH" in
    x86_64)  GO_ARCH="amd64" ;;
    aarch64) GO_ARCH="arm64" ;;
    *)       GO_ARCH="$ARCH" ;;
esac
TARBALL="gpusched_${GPUSCHED_VERSION}_linux_${GO_ARCH}.tar.gz"
DOWNLOAD_URL="https://github.com/${GITHUB_REPO}/releases/download/v${GPUSCHED_VERSION}/${TARBALL}"

if [ -f "$SCRIPT_DIR/gpusched" ] && file "$SCRIPT_DIR/gpusched" | grep -q ELF; then
    cp "$SCRIPT_DIR/gpusched" "$INSTALL_DIR/gpusched"
    chmod 755 "$INSTALL_DIR/gpusched"
    info "Installed from local binary"
else
    info "Downloading $DOWNLOAD_URL"
    TMPDIR=$(mktemp -d)
    if curl -fsSL "$DOWNLOAD_URL" -o "$TMPDIR/$TARBALL"; then
        tar -xzf "$TMPDIR/$TARBALL" -C "$TMPDIR"
        cp "$TMPDIR/gpusched" "$INSTALL_DIR/gpusched"
        chmod 755 "$INSTALL_DIR/gpusched"
        rm -rf "$TMPDIR"
        info "Installed: $INSTALL_DIR/gpusched"
    else
        rm -rf "$TMPDIR"
        error "Download failed. Check that v${GPUSCHED_VERSION} exists at https://github.com/${GITHUB_REPO}/releases"
    fi
fi

# ── Step 6: Create directories and systemd service ───────────────────────────

step 6 "Setting up systemd service"

mkdir -p "$DATA_DIR/snapshots" "$LOG_DIR"

cat > /etc/systemd/system/gpusched.service <<'EOF'
[Unit]
Description=gpusched — GPU Process Manager
After=network.target nvidia-persistenced.service
Wants=nvidia-persistenced.service

[Service]
Type=simple
ExecStart=/usr/local/bin/gpusched daemon --disk-dir /var/lib/gpusched/snapshots
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=gpusched

# Security hardening
NoNewPrivileges=no
ProtectSystem=false

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable gpusched
systemctl start gpusched

sleep 2

if systemctl is-active --quiet gpusched; then
    info "gpusched daemon is running"
else
    warn "Daemon failed to start. Check: journalctl -u gpusched"
fi

# ── Step 7: Verify ───────────────────────────────────────────────────────────

step 7 "Verifying installation"

echo ""
if command -v gpusched &>/dev/null; then
    gpusched status 2>/dev/null && echo "" || warn "Daemon may still be starting up"
fi

echo -e "${GREEN}${BOLD}Installation complete!${NC}"
echo ""
echo -e "  ${BOLD}Quick start:${NC}"
echo "    gpusched status                            # Check GPU and capabilities"
echo "    gpusched run --name my-model -- python3 serve.py"
echo "    gpusched freeze my-model                   # Checkpoint → free GPU"
echo "    gpusched thaw my-model                     # Restore → reclaim GPU"
echo "    gpusched dashboard                         # Interactive dashboard"
echo ""
echo -e "  ${BOLD}Manage the daemon:${NC}"
echo "    sudo systemctl status gpusched"
echo "    sudo journalctl -u gpusched -f"
echo ""
echo -e "  ${BOLD}Docs:${NC} https://github.com/shayonj/gpusched"
echo ""
