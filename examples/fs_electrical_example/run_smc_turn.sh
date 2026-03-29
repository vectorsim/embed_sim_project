#!/usr/bin/env bash
# =============================================================================
#  SMC FOC Workflow Script  -  NANOTEC DB42S02  /  EmbedSim
#  Paul Abraham 2025
#  Linux equivalent of run_smc_turn.bat
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Locate Python -----------------------------------------------------------
if [[ -f ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

# =============================================================================
#  Helpers
# =============================================================================

header() {
    clear
    echo
    echo "  =================================================================="
    echo "    EmbedSim  *  NANOTEC DB42S02  *  SMC FOC Workflow"
    echo "    AURIX TC3xx  *  ISO 26262  *  Paul Abraham  2025"
    echo "  =================================================================="
    echo
}

section_banner() {
    clear
    header
    echo "  =================================================================="
    echo "    $1"
    echo "  =================================================================="
    echo
}

step_banner() {
    echo
    echo "  +------------------------------------------------------------------+"
    printf "  |  STEP %s / %s  --  %s\n" "$1" "$2" "$3"
    echo "  +------------------------------------------------------------------+"
}

confirm_start() {
    local msg="$1"
    printf "  %s (y/n): " "$msg"
    read -r ans
    if [[ "${ans,,}" == "y" ]]; then
        return 0
    fi
    echo "  Cancelled."
    return 1
}

run_timed() {
    echo
    echo "  >> Started  : $(date '+%H:%M:%S')"
    echo "  >> Running  : $*"
    echo
    echo "  ------------------------------------------------------------------"
    local t0
    t0=$(date +%s)
    "$@"
    local err=$?
    local t1
    t1=$(date +%s)
    echo "  ------------------------------------------------------------------"
    echo
    echo "  >> Finished : $(date '+%H:%M:%S')"
    local elapsed=$(( t1 - t0 ))
    local h=$(( elapsed / 3600 ))
    local m=$(( (elapsed % 3600) / 60 ))
    local s=$(( elapsed % 60 ))
    if   (( h > 0 )); then printf "  >> Elapsed  : %dh %dm %ds\n" $h $m $s
    elif (( m > 0 )); then printf "  >> Elapsed  : %dm %ds\n" $m $s
    else                   printf "  >> Elapsed  : %ds\n" $s
    fi
    return $err
}

step_done() {
    local err="$1"
    local label="$2"
    if [[ "$err" == "0" ]]; then
        echo
        echo "  =================================================================="
        echo "    [OK]  $label completed successfully."
        echo "  =================================================================="
        echo
        echo "  Output files written:"
        [[ -f "db42s02_smc_foc_results.png" ]]       && echo "    [+] db42s02_smc_foc_results.png"
        [[ -f "smc_best_gains.json" ]]               && echo "    [+] smc_best_gains.json"
        [[ -f "smc_tuned_gains.json" ]]              && echo "    [+] smc_tuned_gains.json"
        [[ -f "embedsim_gen/embedsim_step.c" ]]      && echo "    [+] embedsim_gen/embedsim_step.c"
        [[ -f "embedsim_gen/embedsim_step.h" ]]      && echo "    [+] embedsim_gen/embedsim_step.h"
        [[ -f "smc_tuner_verify.png" ]]              && echo "    [+] smc_tuner_verify.png"
    else
        echo
        echo "  =================================================================="
        echo "    [ERROR]  $label failed  (exit code $err)"
        echo "  ------------------------------------------------------------------"
        echo "    Troubleshooting:"
        echo "      * Python version:  python3 --version   (need 3.8+)"
        echo "      * Install deps:    pip install -r requirements.txt"
        echo "      * Check fs_electrical_machines/ folder is present"
        echo "      * Option 0: ensure smc_best_gains.json exists first"
        echo "  =================================================================="
    fi
}

finish() {
    echo
    echo "  =================================================================="
    echo "    Script finished."
    echo "  =================================================================="
    echo
}

# =============================================================================
#  MENU
# =============================================================================

header

echo "  Python  : $PYTHON"
echo
echo "  +----------------------------------------------------------------+"
echo "  |                       SELECT MODE                              |"
echo "  +----------------------------------------------------------------+"
echo "  |  T.  SMOKE TEST  (2 min: DE=5 GP=3 t_sim=0.5s 400 RPM)       |  <-- test first"
echo "  |  0.  SIMULATE & CODEGEN only  (load existing gains JSON)      |  <-- small PC"
echo "  |  1.  Complete Workflow         (Tune -> Simulate -> CodeGen)   |"
echo "  |  2.  Quick Test                (20 DE / 15 GP  ~3-5 min)      |"
echo "  |  3.  Production Tuning         (200 DE / 100 GP ~45-60 min)   |"
echo "  |  4.  Standalone Tuner only     (no code generation)           |"
echo "  |  5.  Standalone Tuner + Plot                                   |"
echo "  |  6.  Custom  (interactive)                                     |"
echo "  |  H.  Help                                                      |"
echo "  |  Q.  Quit                                                      |"
echo "  +----------------------------------------------------------------+"
echo
printf "  Enter choice: "
read -r CHOICE

# =============================================================================
#  OPTION T  -  SMOKE TEST
# =============================================================================
if [[ "${CHOICE,,}" == "t" ]]; then
    section_banner "SMOKE TEST  (end-to-end in ~2 minutes)"
    echo "  DE=5 / GP=3 / 400 RPM / t_sim=0.5s / dt=100us"
    echo "  Verifies the full pipeline: tuner -> simulate -> codegen -> plot."
    echo "  Not for production gains -- use to confirm the install works."
    echo
    confirm_start "Start smoke test?" || { finish; exit 0; }
    step_banner 1 3 "DE tuning  (5 iters, fast)"
    run_timed $PYTHON db42s02_tune_simulate_codegen.py \
        --rpm 400 --de_iters 5 --gp_iters 3 --t_sim 0.5 --dt 100e-6
    RESULT=$?
    step_done $RESULT "Smoke test"

# =============================================================================
#  OPTION 0  -  SIMULATE + CODEGEN only
# =============================================================================
elif [[ "$CHOICE" == "0" ]]; then
    section_banner "SIMULATE + CODEGEN  (no tuning)"
    echo "  Loads gains from  smc_best_gains.json  (or header defaults)."
    echo "  Skips the DE/GP optimiser -- fast on any PC."
    echo
    printf "  Target RPM [default 2000]: "; read -r SIM_RPM;  SIM_RPM="${SIM_RPM:-2000}"
    printf "  Simulation time [s, default 2.0]: "; read -r SIM_T; SIM_T="${SIM_T:-2.0}"
    printf "  Time step [us, default 50]: "; read -r SIM_DT_US; SIM_DT_US="${SIM_DT_US:-50}"
    echo
    echo "  +------------------------------------------------------------------+"
    echo "  |  RPM   : $SIM_RPM"
    echo "  |  t_sim : $SIM_T s"
    echo "  |  dt    : ${SIM_DT_US} us  (20 kHz PWM)"
    echo "  +------------------------------------------------------------------+"
    echo
    confirm_start "Start simulate + codegen?" || { finish; exit 0; }
    step_banner 1 2 "Loading gains / preparing simulation"
    run_timed $PYTHON db42s02_tune_simulate_codegen.py \
        --no-tune \
        --rpm "$SIM_RPM" \
        --t_sim "$SIM_T" \
        --dt "${SIM_DT_US}e-6"
    RESULT=$?
    step_done $RESULT "Simulate + CodeGen"

# =============================================================================
#  OPTION 1  -  COMPLETE WORKFLOW
# =============================================================================
elif [[ "$CHOICE" == "1" ]]; then
    section_banner "COMPLETE WORKFLOW  (Tune -> Simulate -> CodeGen)"
    echo "  DE=100 / GP=50 / 2000 RPM / 2.0 s"
    echo "  Expected: 15-20 min on a modern PC"
    echo
    confirm_start "Start complete workflow?" || { finish; exit 0; }
    step_banner 1 3 "DE + GP Bayesian tuning  (100 DE / 50 GP)"
    run_timed $PYTHON db42s02_tune_simulate_codegen.py \
        --rpm 2000 --de_iters 100 --gp_iters 50 --t_sim 2.0
    RESULT=$?
    step_done $RESULT "Complete workflow"

# =============================================================================
#  OPTION 2  -  QUICK TEST
# =============================================================================
elif [[ "$CHOICE" == "2" ]]; then
    section_banner "QUICK TEST  (fast tuning)"
    echo "  DE=20 / GP=15 / 1000 RPM / 1.5 s  (~3-5 min)"
    echo
    confirm_start "Start quick test?" || { finish; exit 0; }
    step_banner 1 3 "DE + GP tuning  (20 DE / 15 GP)"
    run_timed $PYTHON db42s02_tune_simulate_codegen.py \
        --rpm 1000 --de_iters 20 --gp_iters 15 --t_sim 1.5
    RESULT=$?
    step_done $RESULT "Quick test"

# =============================================================================
#  OPTION 3  -  PRODUCTION TUNING
# =============================================================================
elif [[ "$CHOICE" == "3" ]]; then
    section_banner "PRODUCTION TUNING  (maximum quality)"
    echo "  DE=200 / GP=100 / 3000 RPM / 2.0 s"
    echo "  WARNING: ~45-60 minutes -- do not use on a slow PC"
    echo
    confirm_start "Start production tuning?" || { finish; exit 0; }
    step_banner 1 3 "DE global search  (200 iters)"
    run_timed $PYTHON db42s02_tune_simulate_codegen.py \
        --rpm 3000 --de_iters 200 --gp_iters 100 --t_sim 2.0
    RESULT=$?
    step_done $RESULT "Production tuning"

# =============================================================================
#  OPTION 4  -  STANDALONE TUNER
# =============================================================================
elif [[ "$CHOICE" == "4" ]]; then
    section_banner "STANDALONE TUNER  (gains only, no codegen)"
    echo
    confirm_start "Start tuner (DE=80, GP=40, 400 RPM)?" || { finish; exit 0; }
    step_banner 1 1 "Running DE + GP optimiser"
    run_timed $PYTHON smc_fmu_tuner.py \
        --rpm 400 --de_iters 80 --gp_iters 40 --out smc_tuned_gains.json
    RESULT=$?
    step_done $RESULT "Standalone tuner"

# =============================================================================
#  OPTION 5  -  STANDALONE TUNER WITH PLOT
# =============================================================================
elif [[ "$CHOICE" == "5" ]]; then
    section_banner "STANDALONE TUNER + VERIFICATION PLOT"
    echo
    printf "  Target RPM [default 400]: ";   read -r RPM5; RPM5="${RPM5:-400}"
    printf "  DE iters [default 60]: ";      read -r DE5;  DE5="${DE5:-60}"
    printf "  GP iters [default 30]: ";      read -r GP5;  GP5="${GP5:-30}"
    echo
    confirm_start "Start tuner + plot (RPM=$RPM5, DE=$DE5, GP=$GP5)?" || { finish; exit 0; }
    step_banner 1 2 "Running DE + GP optimiser + verification plot"
    run_timed $PYTHON smc_fmu_tuner.py \
        --rpm "$RPM5" --de_iters "$DE5" --gp_iters "$GP5" \
        --verify --out smc_tuned_gains.json
    RESULT=$?
    step_done $RESULT "Tuner + plot"

# =============================================================================
#  OPTION 6  -  CUSTOM
# =============================================================================
elif [[ "$CHOICE" == "6" ]]; then
    section_banner "CUSTOM PARAMETERS"
    echo
    printf "  Target RPM [default 2000]: ";          read -r CRPM;   CRPM="${CRPM:-2000}"
    printf "  DE iters [default 50]: ";              read -r CDE;    CDE="${CDE:-50}"
    printf "  GP iters [default 30]: ";              read -r CGP;    CGP="${CGP:-30}"
    printf "  Simulation time [s, default 2.0]: ";   read -r CTSIM;  CTSIM="${CTSIM:-2.0}"
    printf "  Time step [us, default 50]: ";         read -r CDT_US; CDT_US="${CDT_US:-50}"
    echo
    echo "  +------------------------------------------------------------------+"
    echo "  |  RPM=$CRPM  DE=$CDE  GP=$CGP  t_sim=${CTSIM}s  dt=${CDT_US}us"
    echo "  +------------------------------------------------------------------+"
    echo
    confirm_start "Start with these settings?" || { finish; exit 0; }
    step_banner 1 3 "Custom tuning run"
    run_timed $PYTHON db42s02_tune_simulate_codegen.py \
        --rpm "$CRPM" --de_iters "$CDE" --gp_iters "$CGP" \
        --t_sim "$CTSIM" --dt "${CDT_US}e-6"
    RESULT=$?
    step_done $RESULT "Custom run"

# =============================================================================
#  HELP
# =============================================================================
elif [[ "${CHOICE,,}" == "h" ]]; then
    clear
    header
    echo "  +--------------------------------------------------------------------+"
    echo "  |                   EmbedSim SMC FOC  HELP                          |"
    echo "  +--------------------------------------------------------------------+"
    echo "  |  OPTION T  --  Smoke Test  (~2 minutes, verifies install)         |"
    echo "  |    DE=5 GP=3 t_sim=0.5s dt=100us 400 RPM.                        |"
    echo "  |    Confirms tuner, simulate, codegen and plot all work.           |"
    echo "  |    Gains are NOT production quality -- use Option 1+ for that.   |"
    echo "  |                                                                    |"
    echo "  |  OPTION 0  --  Simulate + CodeGen  (RECOMMENDED for slow PCs)    |"
    echo "  |    Loads smc_best_gains.json (tuned previously).                  |"
    echo "  |    Runs closed-loop simulation at 20 kHz.                         |"
    echo "  |    Emits embedsim_gen/embedsim_step.c/.h for AURIX TC3xx.         |"
    echo "  |    Generates db42s02_smc_foc_results.png.                         |"
    echo "  |    Typical time: 30-120 seconds depending on t_sim.               |"
    echo "  |                                                                    |"
    echo "  |  OPTION 1  --  Complete Workflow  (Tune -> Simulate -> CodeGen)   |"
    echo "  |    Runs DE global search then GP Bayesian refinement.             |"
    echo "  |    Best for first-time commissioning at a new operating point.    |"
    echo "  |                                                                    |"
    echo "  |  ITERATION GUIDE:                                                  |"
    echo "  |    Quick Test   : DE=20,  GP=15   ~3-5 min                        |"
    echo "  |    Balanced     : DE=50,  GP=30   ~10-15 min                      |"
    echo "  |    Production   : DE=100, GP=50   ~20-30 min                      |"
    echo "  |    Maximum      : DE=200, GP=100  ~45-60 min                      |"
    echo "  |                                                                    |"
    echo "  |  COST FUNCTION:                                                    |"
    echo "  |    J = ITAE + 2*overshoot^2 + 0.05*chattering + 0.1*|iq_ss|     |"
    echo "  |                                                                    |"
    echo "  |  OUTPUT FILES:                                                     |"
    echo "  |    smc_best_gains.json           -- optimal SMC gains              |"
    echo "  |    db42s02_smc_foc_results.png   -- 8-panel performance plot       |"
    echo "  |    embedsim_gen/embedsim_step.c  -- AURIX TC3xx C code             |"
    echo "  |    embedsim_gen/embedsim_step.h  -- generated header               |"
    echo "  +--------------------------------------------------------------------+"
    echo
    printf "  Press Enter to continue..."; read -r

# =============================================================================
#  QUIT
# =============================================================================
elif [[ "${CHOICE,,}" == "q" ]]; then
    true   # fall through to finish

else
    echo "  [ERROR] Invalid choice: '$CHOICE'"
fi

finish
