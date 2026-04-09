#!/usr/bin/env bash
# =============================================================================
# algebraic_loop.sh
# EmbedSim — Algebraic Loop Example Runner (Linux / macOS)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

run_script() {
    echo ""
    echo "  Running: $1"
    echo "  --------------------------------------------------------"
    "$PYTHON" "$SCRIPT_DIR/$1"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo ""
        echo "  [ERROR] Script exited with code $rc"
    else
        echo ""
        echo "  [DONE]  Script completed successfully."
    fi
    echo ""
    read -rp "  Press ENTER to return to menu..." _dummy
}

while true; do
    clear
    echo "======================================"
    echo "       EmbedSim Script Runner"
    echo "======================================"
    echo "  1.  example_algebraic_loop.py"
    echo "  2.  Exit"
    echo "======================================"
    read -rp "  Enter choice (1-2): " choice

    case "$choice" in
        1) run_script "example_algebraic_loop.py" ;;
        2) echo ""; echo "  Goodbye."; echo ""; exit 0 ;;
        *) echo ""; echo "  Invalid choice — please enter 1 or 2."; sleep 1 ;;
    esac
done
