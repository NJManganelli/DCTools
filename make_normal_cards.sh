#!/bin/sh
# Build the fiducial datacards (signal = WZ from inc-SR01_IFV, WZ_OFV background
# from inc-SR01_OFV, data-driven DY) for all four eras. Needs the split_FV pickle
# (with inc-SR01_IFV / inc-SR01_OFV filled) referenced by each era config.
python makecard-boost-WZ_asym_FV_pol.py --name WZ --input ./config/inc-WZ/input_UL_2018-WZ_inclusive.yaml    --era 2018  --channel inc-SR01 -rrt datadriven -v dilep_tau_loose_met_hadron_mt
# python make_card.py --name WZfid --input ./config/inc-WZ/input_UL_2017-WZ_inclusive.yaml    --era 2017    --mode fiducial --channel inc-SR01 -rrt datadriven -v dilep_tau_loose_met_hadron_mt
# python make_card.py --name WZfid --input ./config/inc-WZ/input_UL_2016-WZ_inclusive.yaml    --era 2016    --mode fiducial --channel inc-SR01 -rrt datadriven -v dilep_tau_loose_met_hadron_mt
# python make_card.py --name WZfid --input ./config/inc-WZ/input_UL_2016APV-WZ_inclusive.yaml --era 2016APV --mode fiducial --channel inc-SR01 -rrt datadriven -v dilep_tau_loose_met_hadron_mt