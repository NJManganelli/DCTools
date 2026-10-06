#!/usr/bin/env bash 
# W, W+, W-, Z, Z in W+Z, Z in W-Z: 6 cards x 3 templates = the 18 split datasets
for BOSON in W Wp Wm Z Z_WpZ Z_WmZ; do
  # python makecard-boost-WZ_asym_FV_pol.py --name WZpol_${BOSON} --input ./config/inc-WZ/input_UL_2018-WZ_inclusive.yaml    --era 2018    --mode polarization --boson ${BOSON} --channel inc-SR01 -rrt datadriven
  # python makecard-boost-WZ_asym_FV_pol.py --name WZpol_${BOSON} --input ./config/inc-WZ/input_UL_2017-WZ_inclusive.yaml    --era 2017    --mode polarization --boson ${BOSON} --channel inc-SR01 -rrt datadriven
  # python makecard-boost-WZ_asym_FV_pol.py --name WZpol_${BOSON} --input ./config/inc-WZ/input_UL_2016-WZ_inclusive.yaml    --era 2016    --mode polarization --boson ${BOSON} --channel inc-SR01 -rrt datadriven
  # python makecard-boost-WZ_asym_FV_pol.py --name WZpol_${BOSON} --input ./config/inc-WZ/input_UL_2016APV-WZ_inclusive.yaml --era 2016APV --mode polarization --boson ${BOSON} --channel inc-SR01 -rrt datadriven
  python makecard-boost-WZ_asym_FV_pol.py --name WZpol_${BOSON} --input ./config/inc-WZ/input_UL_2024-WZ_inclusive.yaml    --era 2024    --mode polarization --boson ${BOSON} --channel inc-SR01 -rrt datadriven
done