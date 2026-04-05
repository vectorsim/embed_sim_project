#!/usr/bin/env bash
# =============================================================================
# cleanup_all.sh  —  Master cleanup script for Linux
# =============================================================================
# Calls all individual cleanup scripts for:
#   - DFC Controller
#   - SVPWM
#   - SMC Controller
#   - Motor Utility Blocks
#   - Coordinate Transform
# =============================================================================

set -euo pipefail

echo "============================================================"
echo " MASTER CLEANUP - Removing all build artifacts"
echo "============================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Counter for tracking failures
FAILED=0
TOTAL=5

# Colors for output (optional)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# -----------------------------------------------------------------------------
# 1. Clean DFC Controller
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[1/$TOTAL] Cleaning DFC Controller...${NC}"
echo "----------------------------------------"
if [[ -f "$SCRIPT_DIR/cleanup_dfc_controller.sh" ]]; then
    if bash "$SCRIPT_DIR/cleanup_dfc_controller.sh"; then
        echo -e "${GREEN}DFC Controller cleanup completed${NC}"
    else
        echo -e "${RED}[WARNING] DFC Controller cleanup reported errors${NC}"
        ((FAILED++))
    fi
else
    echo -e "${RED}[ERROR] cleanup_dfc_controller.sh not found!${NC}"
    ((FAILED++))
fi
echo ""

# -----------------------------------------------------------------------------
# 2. Clean SVPWM
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[2/$TOTAL] Cleaning SVPWM...${NC}"
echo "----------------------------------------"
if [[ -f "$SCRIPT_DIR/cleanup_svpwm.sh" ]]; then
    if bash "$SCRIPT_DIR/cleanup_svpwm.sh"; then
        echo -e "${GREEN}SVPWM cleanup completed${NC}"
    else
        echo -e "${RED}[WARNING] SVPWM cleanup reported errors${NC}"
        ((FAILED++))
    fi
else
    echo -e "${RED}[ERROR] cleanup_svpwm.sh not found!${NC}"
    ((FAILED++))
fi
echo ""

# -----------------------------------------------------------------------------
# 3. Clean SMC Controller
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[3/$TOTAL] Cleaning SMC Controller...${NC}"
echo "----------------------------------------"
if [[ -f "$SCRIPT_DIR/cleanup_smc_controller.sh" ]]; then
    if bash "$SCRIPT_DIR/cleanup_smc_controller.sh"; then
        echo -e "${GREEN}SMC Controller cleanup completed${NC}"
    else
        echo -e "${RED}[WARNING] SMC Controller cleanup reported errors${NC}"
        ((FAILED++))
    fi
else
    echo -e "${RED}[ERROR] cleanup_smc_controller.sh not found!${NC}"
    ((FAILED++))
fi
echo ""

# -----------------------------------------------------------------------------
# 4. Clean Motor Utility Blocks
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[4/$TOTAL] Cleaning Motor Utility Blocks...${NC}"
echo "----------------------------------------"
if [[ -f "$SCRIPT_DIR/cleanup_motor_utility_blocks.sh" ]]; then
    if bash "$SCRIPT_DIR/cleanup_motor_utility_blocks.sh"; then
        echo -e "${GREEN}Motor Utility Blocks cleanup completed${NC}"
    else
        echo -e "${RED}[WARNING] Motor Utility Blocks cleanup reported errors${NC}"
        ((FAILED++))
    fi
else
    echo -e "${RED}[ERROR] cleanup_motor_utility_blocks.sh not found!${NC}"
    ((FAILED++))
fi
echo ""

# -----------------------------------------------------------------------------
# 5. Clean Coordinate Transform
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[5/$TOTAL] Cleaning Coordinate Transform...${NC}"
echo "----------------------------------------"
if [[ -f "$SCRIPT_DIR/cleanup_coordinate_transform.sh" ]]; then
    if bash "$SCRIPT_DIR/cleanup_coordinate_transform.sh"; then
        echo -e "${GREEN}Coordinate Transform cleanup completed${NC}"
    else
        echo -e "${RED}[WARNING] Coordinate Transform cleanup reported errors${NC}"
        ((FAILED++))
    fi
else
    echo -e "${RED}[ERROR] cleanup_coordinate_transform.sh not found!${NC}"
    ((FAILED++))
fi
echo ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo "============================================================"
echo " MASTER CLEANUP COMPLETE"
echo "============================================================"
echo ""
if [[ $FAILED -eq 0 ]]; then
    echo -e "${GREEN}Status: All $TOTAL cleanups completed successfully!${NC}"
    echo ""
    echo "All build artifacts have been removed from:"
    echo "  - DFC Controller"
    echo "  - SVPWM"
    echo "  - SMC Controller"
    echo "  - Motor Utility Blocks"
    echo "  - Coordinate Transform"
else
    echo -e "${RED}Status: $FAILED of $TOTAL cleanups had issues.${NC}"
    echo "Please check the output above for details."
fi
echo ""
echo "============================================================"