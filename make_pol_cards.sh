#!/usr/bin/env bash
# set -euo pipefail

# CONFIG="config/inc-WZ/input_UL_2018-WZ_inclusive.yaml"
# CHANNEL="inc-SR01"
# ERA="2018"
# RRT="datadriven"                         
# ALL_BOSONS=(W Wp Wm Z)

# case "${1:-all}" in
#   ""|all)     BOSONS=("${ALL_BOSONS[@]}") ;;
#   W|Wp|Wm|Z)  BOSONS=("$1") ;;
#   *) echo "unknown arg '$1' (use: all | W|Wp|Wm|Z)"; exit 1 ;;
# esac

# for B in "${BOSONS[@]}"; do
#   echo "============================================================"
#   echo "  building polarization card : boson=${B}  (DY = data-driven)"
#   echo "============================================================"
#   python make_card.py \
#       -i "${CONFIG}" \
#       --mode polarization --boson "${B}" \
#       -c "${CHANNEL}" -y "${ERA}" \
#       -rrt "${RRT}" \
#       -n "WZpol_${B}"
#   echo "  -> cards-WZpol_${B}/shapes-${CHANNEL}${ERA}.dat"
# done

# echo "done. (look for 'Overwriting: ... DY replaced by DY' above = data-driven DY active)"
 
for BOSON in W Wp Wm Z; do
  python make_card.py --name WZpol_${BOSON} --input ./config/inc-WZ/input_UL_2018-WZ_inclusive.yaml    --era 2018    --mode polarization --boson ${BOSON} --channel inc-SR01 -rrt datadriven
  python make_card.py --name WZpol_${BOSON} --input ./config/inc-WZ/input_UL_2017-WZ_inclusive.yaml    --era 2017    --mode polarization --boson ${BOSON} --channel inc-SR01 -rrt datadriven
  python make_card.py --name WZpol_${BOSON} --input ./config/inc-WZ/input_UL_2016-WZ_inclusive.yaml    --era 2016    --mode polarization --boson ${BOSON} --channel inc-SR01 -rrt datadriven
  python make_card.py --name WZpol_${BOSON} --input ./config/inc-WZ/input_UL_2016APV-WZ_inclusive.yaml --era 2016APV --mode polarization --boson ${BOSON} --channel inc-SR01 -rrt datadriven
done