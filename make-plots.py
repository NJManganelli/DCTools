import argparse
import gzip
import pickle
import matplotlib.pyplot as plt
import yaml
import io
import os
import numpy as np
import hist
from typing import Any, IO, Dict, List, Iterable
import dctools
from dctools import plot as plotter
from dctools.plot import plotting
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import mplhep as hep
from scipy import stats as st
np.seterr(all='warn')
plt.ioff()

from dctools import dict_to_hist_axis

#config_2016APV = dctools.read_config("config/input_UL_2016APV-amalfi.yaml")
#config_2016    = dctools.read_config("config/input_UL_2016-amalfi.yaml")
#config_2017    = dctools.read_config("config/input_UL_2017-amalfi.yaml")
#config_2018    = dctools.read_config("config/input_UL_2018-amalfi.yaml")


#for channel in config_2018.plotting:
#    ch_cfg = config_2018.plotting[channel]
#    # if channel not in ['vbs-SR', 'vbs-TT', 'vbs-DY', 'vbs-EM', 'vbs-3L']: continue
#    for vname in ch_cfg:
#        # if 'met_pt' not in vname: continue
#        v_cfg = ch_cfg[vname]
#        plotting(
#            config_2018, vname, channel,
#            rebin=v_cfg.rebin,
#            xlim=v_cfg.range,
#            blind=v_cfg.blind,
#            era="2018"
#        )
#        plotting(
#            config_2017, vname, channel,
#            rebin=v_cfg.rebin,
#            xlim=v_cfg.range,
#            blind=v_cfg.blind,
#            era="2017"
#        )
#        plotting(
#            config_2016, vname, channel,
#            rebin=v_cfg.rebin,
#            xlim=v_cfg.range,
#            blind=v_cfg.blind,
#            era="2016"
#        )
#        plotting(
#            config_2016APV, vname, channel,
#            rebin=v_cfg.rebin,
#            xlim=v_cfg.range,
#            blind=v_cfg.blind,
#            era="2016APV"
#        )
def main():
    parser = argparse.ArgumentParser(description='DCTools Plotting Tool')
    parser.add_argument("-i"  , "--input"   , type=str , default="./config/input_UL_2018_timgad-vbs.yaml")
    parser.add_argument("-y"  , "--era"     , type=str , default='2018')
    parser.add_argument("-v"  , "--variables", nargs="*", type=str)
    parser.add_argument("-c"  , "--channels" , nargs='*', type=str)
    parser.add_argument("-rrt", "--remap_replacement_types", nargs='*', type=str, default=[])
    parser.add_argument("-gbwn", "--global_bin_width_norm", type=float, default=None, const=1.0, nargs="?", help="if set, normalize all histograms to `bin_width_norm / bin_width`, suggested (const) bin_width_norm value is 1.0")
    parser.add_argument("--no_ratios", action="store_true", help="disable ratio panel in the plots")
    parser.add_argument("-b"  , "--blindings" , nargs='*', type=bool, default=[False])
    parser.add_argument('--checksyst', action='store_true')
    parser.add_argument('-cf', "--combine_fit", type=str, default="pre-combine")
    parser.add_argument('-ctu', "--combine_total_uncertainty", type=str, default="total_background")
    parser.add_argument('-ccg', "--combine_channel_groups", nargs="*", type=str, default=None)
    parser.add_argument('-ccgb', "--combine_channel_group_blindings", nargs="*", type=bool, default=[False])
    parser.add_argument('--logx', action="store_true", help="make x axis in log scale")

    options = parser.parse_args()
    config = dctools.read_config(options.input)

    if options.combine_fit in ["prefit", "fit_b", "fit_s"]:
        if options.logx:
            raise NotImplementedError("logx not yet implemented for combine plotting")
        # for combine plotting, for the prefit, background-only fit, or signal + background fit
        do_channels = (isinstance(options.channels, list) and len(options.channels) > 0)
        do_groups = (isinstance(options.combine_channel_groups, list) and len(options.combine_channel_groups) > 0)
        at_least_one = do_channels or do_groups
        assert at_least_one, "In combine output mode, must have a channel or at least one combine_channel_group name"
        assert options.variables, "In combine output mode, must specify variables to be plotted"

        #plot individual channels
        if options.channels:
            iter_blindings = options.blindings
            if len(iter_blindings) == 1 and len(options.channels) > 1:
                iter_blindings = iter_blindings * len(options.channels)
            for channel, blind in zip(options.channels,iter_blindings):
                for vname in options.variables:
                    _ = plotting(config, vname, channel,
                                 rebin = 1,
                                 xlim = [],
                                 blind = blind,
                                 era = options.era,
                                 remap_replacement_types = options.remap_replacement_types,
                                 logx=options.logx,
                                 bin_width_norm = options.global_bin_width_norm,
                                 no_ratios = options.no_ratios,
                                 checksyst = False,
                                 combine_fit = options.combine_fit,
                                 combine_total_uncertainty = options.combine_total_uncertainty,
                                 combine_channel_group = None,
                                )

        #plot stacks of channels
        if options.combine_channel_groups:
            group_blindings = options.combine_channel_group_blindings
            if len(group_blindings) == 1 and len(options.combine_channel_groups) > 1:
                group_blindings = group_blindings * len(options.combine_channel_groups)
            for channel_group, group_blind in zip(options.combine_channel_groups,group_blindings):
                for vname in options.variables:
                    _ = plotting(config, vname, None,
                                 rebin = 1,
                                 xlim = [],
                                 blind = group_blind,
                                 #era = options.era, #picked up from configuration automatically
                                 remap_replacement_types = options.remap_replacement_types, # may not be needed/used
                                 logx=options.logx,
                                 bin_width_norm = options.global_bin_width_norm,
                                 no_ratios = options.no_ratios,
                                 checksyst = False,
                                 combine_fit = options.combine_fit,
                                 combine_total_uncertainty = options.combine_total_uncertainty,
                                 combine_channel_group = channel_group,
                                )

    elif options.combine_fit == "pre-combine":
        for channel in config.plotting:
            if options.channels and len(options.channels) > 0 and channel not in options.channels:
                continue
            ch_cfg = config.plotting[channel]
            for vname in ch_cfg:
                if options.variables and len(options.variables) > 0 and vname not in options.variables:
                    continue
                v_cfg = ch_cfg[vname]
                config_logx = v_cfg.logx if "logx" in v_cfg else False
                if options.logx :
                    config_logx = True #if the command line option is true overwrite it
                _ = plotting(config, vname, channel,
                             rebin = v_cfg.rebin,
                             xlim = v_cfg.range,
                             blind = v_cfg.blind,
                             era = options.era,
                             remap_replacement_types = options.remap_replacement_types,
                             logx=config_logx,
                             bin_width_norm = options.global_bin_width_norm,
                             no_ratios = options.no_ratios,
                             checksyst = options.checksyst,
                             combine_fit = options.combine_fit,
                            )
    else:
        raise ValueError(f"Invalid combine_fit option {options.combine_fit}, please choose from 'pre-combine' (boost) or 'prefit'/'fit_b'/'fit_s' (combine)")

if __name__ == "__main__":
    main()
