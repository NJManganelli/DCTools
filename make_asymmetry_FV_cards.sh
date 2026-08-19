for CH in p m; do
  for ERA in 2016 2016APV 2017 2018; do
    python make_card_asym_FV.py --name WZasymfid_${CH} \
        --input ./config/inc-WZ/input_UL_${ERA}-WZ_inclusive.yaml --era ${ERA} \
        --mode asymmetry_fiducial --channel inc-SR01_${CH} \
        -v dilep_tau_loose_met_hadron_mt -rrt datadriven
  done
done