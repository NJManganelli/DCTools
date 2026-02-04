#!/bin/sh
python makecard-boost-WZ.py --name WZ_had --input ./config/inc-WZ/input_UL_2018-WZ_inclusive.yaml --era 2018 --variable dilep_tau_loose_met_hadron_mt --channel inc-SR0 -rrt datadriven
python makecard-boost-WZ.py --name WZ_had --input ./config/inc-WZ/input_UL_2018-WZ_inclusive.yaml --era 2018 --variable dilep_tau_loose_met_hadron_mt --channel inc-SR1 -rrt datadriven
python makecard-boost-WZ.py --name WZ_had --input ./config/inc-WZ/input_UL_2018-WZ_inclusive.yaml --era 2018 --variable dilep_tau_loose_met_hadron_mt --channel inc-DY0 
python makecard-boost-WZ.py --name WZ_had --input ./config/inc-WZ/input_UL_2018-WZ_inclusive.yaml --era 2018 --variable dilep_tau_loose_met_hadron_mt --channel inc-DY1 

python makecard-boost-WZ.py --name WZ_had --input ./config/inc-WZ/input_UL_2017-WZ_inclusive.yaml --era 2017 --variable dilep_tau_loose_met_hadron_mt --channel inc-SR0 -rrt datadriven
python makecard-boost-WZ.py --name WZ_had --input ./config/inc-WZ/input_UL_2017-WZ_inclusive.yaml --era 2017 --variable dilep_tau_loose_met_hadron_mt --channel inc-SR1 -rrt datadriven
python makecard-boost-WZ.py --name WZ_had --input ./config/inc-WZ/input_UL_2017-WZ_inclusive.yaml --era 2017 --variable dilep_tau_loose_met_hadron_mt --channel inc-DY0 
python makecard-boost-WZ.py --name WZ_had --input ./config/inc-WZ/input_UL_2017-WZ_inclusive.yaml --era 2017 --variable dilep_tau_loose_met_hadron_mt --channel inc-DY1 

python makecard-boost-WZ.py --name WZ_had --input ./config/inc-WZ/input_UL_2016-WZ_inclusive.yaml --era 2016 --variable dilep_tau_loose_met_hadron_mt --channel inc-SR0 -rrt datadriven
python makecard-boost-WZ.py --name WZ_had --input ./config/inc-WZ/input_UL_2016-WZ_inclusive.yaml --era 2016 --variable dilep_tau_loose_met_hadron_mt --channel inc-SR1 -rrt datadriven
python makecard-boost-WZ.py --name WZ_had --input ./config/inc-WZ/input_UL_2016-WZ_inclusive.yaml --era 2016 --variable dilep_tau_loose_met_hadron_mt --channel inc-DY0 
python makecard-boost-WZ.py --name WZ_had --input ./config/inc-WZ/input_UL_2016-WZ_inclusive.yaml --era 2016 --variable dilep_tau_loose_met_hadron_mt --channel inc-DY1 

python makecard-boost-WZ.py --name WZ_had --input ./config/inc-WZ/input_UL_2016APV-WZ_inclusive.yaml --era 2016APV --variable dilep_tau_loose_met_hadron_mt --channel inc-SR0 -rrt datadriven
python makecard-boost-WZ.py --name WZ_had --input ./config/inc-WZ/input_UL_2016APV-WZ_inclusive.yaml --era 2016APV --variable dilep_tau_loose_met_hadron_mt --channel inc-SR1 -rrt datadriven
python makecard-boost-WZ.py --name WZ_had --input ./config/inc-WZ/input_UL_2016APV-WZ_inclusive.yaml --era 2016APV --variable dilep_tau_loose_met_hadron_mt --channel inc-DY0 
python makecard-boost-WZ.py --name WZ_had --input ./config/inc-WZ/input_UL_2016APV-WZ_inclusive.yaml --era 2016APV --variable dilep_tau_loose_met_hadron_mt --channel inc-DY1 
