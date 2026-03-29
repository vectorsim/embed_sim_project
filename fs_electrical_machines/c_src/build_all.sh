#!/usr/bin/env bash
# =============================================================================
# build_all.sh  —  fs_electrical_machines/c_src
# =============================================================================
# Build all EmbedSim Cython extensions in correct dependency order:
#
#   1. coordinate_transform  (embed_sim_matrix.c + embed_sim_coordinate_transform.c)
#   2. motor_utility_blocks  (embed_sim_motor_utility_blocks.c)
#   3. smc_controller        (embed_sim_smc_controller.c + coord transform + matrix)
#   4. svpwm                 (embed_sim_sv_pwm.c + matrix)
#
# Usage:
#   cd fs_electrical_machines/c_src
#   chmod +x build_all.sh
#   ./build_all.sh
#
# Pass --clean-only to just wipe all build artefacts without compiling.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CLEAN_ONLY=0
for arg in "$@"; do
    [[ "$arg" == "--clean-only" ]] && CLEAN_ONLY=1
done

# ---------------------------------------------------------------------------
# Colour helpers (degrade gracefully if tput unavailable)
# ---------------------------------------------------------------------------
if command -v tput &>/dev/null && tput setaf 1 &>/dev/null; then
    RED=$(tput setaf 1); GREEN=$(tput setaf 2); YELLOW=$(tput setaf 3)
    CYAN=$(tput setaf 6); BOLD=$(tput bold); RESET=$(tput sgr0)
else
    RED=""; GREEN=""; YELLOW=""; CYAN=""; BOLD=""; RESET=""
fi

banner() { echo ""; echo "${BOLD}${CYAN}══════════════════════════════════════════════════${RESET}"; \
           echo "${BOLD}${CYAN}  $1${RESET}"; \
           echo "${BOLD}${CYAN}══════════════════════════════════════════════════${RESET}"; }
ok()     { echo "  ${GREEN}✔${RESET}  $1"; }
warn()   { echo "  ${YELLOW}⚠${RESET}  $1"; }
fail()   { echo "  ${RED}✘${RESET}  $1"; }

# ---------------------------------------------------------------------------
# Build targets — name : script
# ---------------------------------------------------------------------------
declare -a TARGETS=(
    "coordinate_transform:build_coordinate_transform.sh"
    "motor_utility_blocks:build_motor_utility_blocks.sh"
    "smc_controller:build_smc_controller.sh"
    "svpwm:build_svpwm.sh"
)

# ---------------------------------------------------------------------------
banner "EmbedSim — build all Cython extensions"
echo ""
echo "  Script dir : $SCRIPT_DIR"
echo "  Mode       : $([ $CLEAN_ONLY -eq 1 ] && echo 'clean only' || echo 'full build')"
echo ""

if [ $CLEAN_ONLY -eq 1 ]; then
    echo "Cleaning all build artefacts..."
    rm -rf build
    rm -f ./*.so ./*.cpython*.so
    PARENT_DIR="$(dirname "$SCRIPT_DIR")"
    rm -f "$PARENT_DIR"/*.so
    ok "All artefacts removed."
    exit 0
fi

# ---------------------------------------------------------------------------
# Verify each sub-script exists before starting
# ---------------------------------------------------------------------------
for entry in "${TARGETS[@]}"; do
    name="${entry%%:*}"
    script="${entry##*:}"
    if [ ! -f "$SCRIPT_DIR/$script" ]; then
        fail "Missing build script: $script"
        echo ""
        echo "  Expected location: $SCRIPT_DIR/$script"
        echo "  Ensure all build_*.sh scripts are present in c_src/ before running build_all.sh."
        exit 1
    fi
    chmod +x "$SCRIPT_DIR/$script"
done

# ---------------------------------------------------------------------------
# Run each build script, track results
# ---------------------------------------------------------------------------
declare -a PASSED=()
declare -a FAILED=()

for entry in "${TARGETS[@]}"; do
    name="${entry%%:*}"
    script="${entry##*:}"

    banner "Building: $name"
    set +e
    bash "$SCRIPT_DIR/$script"
    STATUS=$?
    set -e

    if [ $STATUS -eq 0 ]; then
        ok "$name — built successfully"
        PASSED+=("$name")
    else
        fail "$name — build FAILED (exit $STATUS)"
        FAILED+=("$name")
        echo ""
        warn "Continuing with remaining targets..."
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
banner "Build summary"
echo ""
if [ ${#PASSED[@]} -gt 0 ]; then
    for name in "${PASSED[@]}"; do
        ok "$name"
    done
fi
if [ ${#FAILED[@]} -gt 0 ]; then
    for name in "${FAILED[@]}"; do
        fail "$name"
    done
fi
echo ""

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "${RED}${BOLD}${#FAILED[@]} target(s) failed.${RESET}"
    echo "Re-run the individual build_*.sh script to see the full error."
    exit 1
else
    echo "${GREEN}${BOLD}All ${#PASSED[@]} targets built successfully.${RESET}"
    echo ""
    echo "Quick import check:"
    echo "  python3 -c \""
    echo "    from fs_electrical_machines.coordinate_transform_wrapper import EmbedSimCoordinateTransform"
    echo "    from fs_electrical_machines.smc_controller_wrapper      import SMCControllerWrapper"
    echo "    from fs_electrical_machines.svpwm_wrapper               import EmbedSimSVPWM"
    echo "    from motor_utility_blocks import SpeedRampBlock"
    echo "    print('All imports OK')"
    echo "  \""
    echo ""
fi
