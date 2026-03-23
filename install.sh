#!/bin/sh
# modl installer — downloads the latest release binary for your platform
# Usage: curl -fsSL https://modl.run/install.sh | sh
#
# Options:
#   --quick              Non-interactive: init + pull model + install service
#   --model <id>         Model to pull (default: flux-schnell, used with --quick)
#   --no-service         Skip service installation (used with --quick)
#   MODL_INSTALL_DIR     Override binary install path (default: /usr/local/bin)

set -e

REPO="modl-org/modl"
INSTALL_DIR="${MODL_INSTALL_DIR:-/usr/local/bin}"
QUICK=0
MODEL="flux-schnell"
NO_SERVICE=0

# Parse flags
while [ $# -gt 0 ]; do
    case "$1" in
        --quick)    QUICK=1; shift ;;
        --model)    MODEL="$2"; shift 2 ;;
        --no-service) NO_SERVICE=1; shift ;;
        *)          echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Check for required tools
for cmd in curl tar; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: '$cmd' is required but not installed."
        exit 1
    fi
done

# Detect OS and architecture
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux)
        case "$ARCH" in
            x86_64)  TARGET="x86_64-unknown-linux-gnu" ;;
            aarch64) TARGET="aarch64-unknown-linux-gnu" ;;
            *)       echo "Unsupported architecture: $ARCH"; exit 1 ;;
        esac
        ;;
    Darwin)
        case "$ARCH" in
            x86_64)  TARGET="x86_64-apple-darwin" ;;
            arm64)   TARGET="aarch64-apple-darwin" ;;
            *)       echo "Unsupported architecture: $ARCH"; exit 1 ;;
        esac
        ;;
    *)
        echo "Unsupported OS: $OS (use Windows installer or cargo install modl)"
        exit 1
        ;;
esac

# Get latest release tag
echo "Detecting latest version..."
LATEST=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" | grep '"tag_name"' | head -1 | cut -d'"' -f4)

if [ -z "$LATEST" ]; then
    echo "Could not determine latest version."
    echo "Check your network connection, or install from source:"
    echo "  cargo install --git https://github.com/${REPO}"
    exit 1
fi

echo "Installing modl ${LATEST} for ${TARGET}..."

# Download binary and checksums
ASSET="modl-${LATEST}-${TARGET}.tar.gz"
URL="https://github.com/${REPO}/releases/download/${LATEST}/${ASSET}"
CHECKSUM_URL="https://github.com/${REPO}/releases/download/${LATEST}/SHA256SUMS"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

curl -fsSL "$URL" -o "$TMPDIR/${ASSET}"

# Verify SHA256 checksum if SHA256SUMS is available
if curl -fsSL "$CHECKSUM_URL" -o "$TMPDIR/SHA256SUMS" 2>/dev/null; then
    EXPECTED=$(grep "$ASSET" "$TMPDIR/SHA256SUMS" | awk '{print $1}')
    if [ -n "$EXPECTED" ]; then
        if command -v sha256sum >/dev/null 2>&1; then
            ACTUAL=$(sha256sum "$TMPDIR/${ASSET}" | awk '{print $1}')
        elif command -v shasum >/dev/null 2>&1; then
            ACTUAL=$(shasum -a 256 "$TMPDIR/${ASSET}" | awk '{print $1}')
        else
            echo "Warning: No sha256sum or shasum found — skipping checksum verification."
            ACTUAL=""
            EXPECTED=""
        fi

        if [ -n "$EXPECTED" ] && [ "$ACTUAL" != "$EXPECTED" ]; then
            echo "Checksum verification failed!"
            echo "  Expected: $EXPECTED"
            echo "  Got:      $ACTUAL"
            echo "The download may be corrupted. Please try again."
            exit 1
        elif [ -n "$EXPECTED" ]; then
            echo "Checksum verified."
        fi
    fi
else
    echo "Note: No SHA256SUMS found for this release — skipping verification."
fi

tar xzf "$TMPDIR/${ASSET}" -C "$TMPDIR"

# Install binary
if [ -w "$INSTALL_DIR" ]; then
    mv "$TMPDIR/modl" "$INSTALL_DIR/modl"
else
    echo "Installing to $INSTALL_DIR (requires sudo)..."
    sudo mv "$TMPDIR/modl" "$INSTALL_DIR/modl"
fi

chmod +x "$INSTALL_DIR/modl"

# Install Python worker (next to binary, where the CLI expects it)
if [ -d "$TMPDIR/python" ]; then
    PYTHON_DIR="$INSTALL_DIR/python"
    if [ -w "$INSTALL_DIR" ]; then
        rm -rf "$PYTHON_DIR"
        mv "$TMPDIR/python" "$PYTHON_DIR"
    else
        sudo rm -rf "$PYTHON_DIR"
        sudo mv "$TMPDIR/python" "$PYTHON_DIR"
    fi
fi

echo ""
echo "modl ${LATEST} installed to ${INSTALL_DIR}/modl"

# Quick install mode: init + pull + service
if [ "$QUICK" = "1" ]; then
    echo ""
    echo "Running quick setup..."

    # Non-interactive init
    "$INSTALL_DIR/modl" init --defaults

    # Pull the specified model
    echo ""
    echo "Pulling model: ${MODEL}"
    "$INSTALL_DIR/modl" pull "$MODEL"

    # Install service (unless --no-service)
    if [ "$NO_SERVICE" = "0" ]; then
        echo ""
        "$INSTALL_DIR/modl" serve --install-service
    fi

    echo ""
    echo "Quick setup complete!"
    echo "  Web UI: http://localhost:3333"
    echo "  Model:  ${MODEL}"
else
    echo ""
    echo "Get started:"
    echo "  modl init              # Configure your setup"
    echo "  modl pull flux-dev     # Install a model"
    echo ""
    echo "Docs: https://modl.run/docs"
    echo ""
fi
