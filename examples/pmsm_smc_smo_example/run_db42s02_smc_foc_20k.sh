#!/usr/bin/env bash
# =============================================================================
# run_db42s02_smc_foc_20k.sh
# EmbedSim -- DB42S02 SMC FOC closed-loop simulation
# NANOTEC DB42S02 PMSM  |  AURIX TC3xx 20 kHz
#
# Usage:
#   ./run_db42s02_smc_foc_20k.sh [--no-tune] [--no-anim]
#
# Options:
#   --no-tune   Skip the interactive gain-tuner prompt (pass 'n' automatically)
#   --no-anim   Skip the interactive animation prompt (pass 'n' automatically)
#
# Both flags together give a fully non-interactive run:
#   ./run_db42s02_smc_foc_20k.sh --no-tune --no-anim
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Locate project root via .project_root_marker
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_root() {
    local dir="$1"
    while [[ "$dir" != "/" ]]; do
        if [[ -f "$dir/.project_root_marker" ]]; then
            echo "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    echo ""
}

PROJECT_ROOT="$(find_root "$SCRIPT_DIR")"
if [[ -z "$PROJECT_ROOT" ]]; then
    echo "ERROR: .project_root_marker not found in any parent of $SCRIPT_DIR"
    exit 1
fi

EXAMPLE_DIR="$PROJECT_ROOT/pmsm_smc_smo_example"
SCRIPT="$EXAMPLE_DIR/db42s02_closed_loop_smc_foc_20k.py"

if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: Script not found: $SCRIPT"
    exit 1
fi

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
NO_TUNE=0
NO_ANIM=0

for arg in "$@"; do
    case "$arg" in
        --no-tune) NO_TUNE=1 ;;
        --no-anim) NO_ANIM=1 ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--no-tune] [--no-anim]"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Build the stdin pipe for the two interactive prompts
# ---------------------------------------------------------------------------
TUNE_ANSWER="y"
ANIM_ANSWER="y"
[[ $NO_TUNE -eq 1 ]] && TUNE_ANSWER="n"
[[ $NO_ANIM -eq 1 ]] && ANIM_ANSWER="n"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  EmbedSim -- DB42S02 SMC FOC 20 kHz"
echo "  Project root : $PROJECT_ROOT"
echo "  Script       : $SCRIPT"
echo "  Gain tuner   : $( [[ $NO_TUNE -eq 1 ]] && echo "SKIP" || echo "interactive" )"
echo "  Animation    : $( [[ $NO_ANIM -eq 1 ]] && echo "SKIP" || echo "interactive" )"
echo "============================================================"

cd "$EXAMPLE_DIR"

printf '%s\n%s\n' "$TUNE_ANSWER" "$ANIM_ANSWER" \
    | python db42s02_closed_loop_smc_foc_20k.py

echo ""
echo "============================================================"
echo "  Done."
echo "  Results in: $EXAMPLE_DIR"
echo "  AURIX code: $PROJECT_ROOT/embedsim_gen/"
echo "============================================================"
