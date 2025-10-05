from __future__ import division

import hist
import os 
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import mplhep as hep
from typing import Any, List, Iterable
from hist.intervals import ratio_uncertainty
from . import datagroup
from . import plot as plotter
from cycler import cycler
from scipy import interpolate

_plot_colors_ = [
    '#99B6F7','#46B3A5',
    '#F6D68D','#2E6D92',
    '#580C82','#8E31A1',
    '#FE4773',
]

mlp.rcParams['axes.grid'          ] =  False
mlp.rcParams['axes.labelsize'     ] =  16
mlp.rcParams['axes.linewidth'     ] =  1.4     ## edge linewidth
mlp.rcParams['legend.framealpha'  ] =  0
mlp.rcParams['legend.fancybox'    ] =  False
mlp.rcParams['legend.frameon'     ] =  False
mlp.rcParams['xtick.top'          ] =  True
mlp.rcParams['xtick.labelsize'    ] =  14
mlp.rcParams['xtick.direction'    ] =  'in'     ## direction: in, out, or inout
mlp.rcParams['xtick.minor.visible'] =  True   ## visibility of minor ticks on x-axis
mlp.rcParams['xtick.major.size'   ] =  4.5    ## major tick size in points
mlp.rcParams['xtick.minor.size'   ] =  3      ## minor tick size in points
mlp.rcParams['xtick.major.width'  ] =  1.0    ## major tick width in points
mlp.rcParams['xtick.minor.width'  ] =  0.8    ## minor tick width in points
mlp.rcParams['ytick.right'        ] =  True
mlp.rcParams['ytick.labelsize'    ] =  14
mlp.rcParams['ytick.direction'    ] =  'in'     ## direction: in, out, or inout
mlp.rcParams['ytick.minor.visible'] =  True   ## visibility of minor ticks on x-axis
mlp.rcParams['ytick.major.size'   ] =  4.5    ## major tick size in points
mlp.rcParams['ytick.minor.size'   ] =  3      ## minor tick size in points
mlp.rcParams['ytick.major.width'  ] =  1.0    ## major tick width in points
mlp.rcParams['ytick.minor.width'  ] =  0.8    ## minor tick width in points
mlp.rcParams['axes.prop_cycle'] = cycler(color=_plot_colors_)


def fill_with_interpolation_1d(hview):
    '''
    interpolate to fill nan values
    '''
    inds = np.arange(hview.value.shape[0])
    mask = (
        np.isfinite(hview.value) & 
        np.isfinite(hview.variance) & 
        (np.abs(hview.value)    < 1e30) & 
        (np.abs(hview.variance) < 1e30) &
        (hview.value >= 0)
    )
    good = np.where(mask)
    
    fval = interpolate.interp1d(inds[good], hview.value[good],bounds_error=False)
    fvar = interpolate.interp1d(inds[good], hview.variance[good],bounds_error=False)
    new_val = np.where(mask, hview.value   ,fval(inds))
    new_var = np.where(mask, hview.variance,fvar(inds))
    # if np.any(~mask):
    #     print("not good : ", hview.variance[~mask], "changed  : ", new_var[~mask])
    return new_val, new_var


def add_process_axis(
        histograms: dict[str, datagroup] | dict[str, hist.Hist], 
        axis_name: str = 'process',
        flow: bool = True) -> hist.Hist:

    storage = None
    histos = {}
    for it, (n, p) in enumerate(histograms.items()):
        if isinstance(p, hist.Hist):
            _h = p
        elif isinstance(p, datagroup):
            _h = p.to_boost()
        else:
            raise ValueError("not recongnised type")

        if len(_h.shape) == 0:
            continue
        # print(n, _h.view(flow=flow).shape)
        if storage is None:
            axes = [axis for axis in _h.axes]
            storage = _h._storage_type()
        histos[n] = _h
            
    iterator = histos.keys()
    new_axis = hist.axis.StrCategory(iterator, name=axis_name, label=axis_name)
    axes.insert(0, new_axis)
    new_hist = hist.hist.Hist(
        *axes,
        storage,
    )
    
    filled_keys = set()
    for key, val in histos.items():
        idx = new_axis.index(key)
        if idx in filled_keys:
            raise ValueError(f"Duplicate key found for Variable type: {key} -> {float(key)}")
        else: 
            filled_keys.add(idx)
        for sys in new_hist.axes["systematic"]:
            ids_new = new_hist.axes["systematic"].index(sys)
            if sys in list(val.axes["systematic"]):
                ids_val = val.axes["systematic"].index(sys)
                hview = val.view(flow=flow)[ids_val,...]
                hview.value, hview.variance = fill_with_interpolation_1d(hview)
                new_hist.view(flow=flow)[idx,ids_new,...] = hview
            else:
                # use nominal in case the systematic doesn't exist
                ids_nom = val.axes["systematic"].index("nominal")
                hview = val.view(flow=flow)[ids_nom,...]
                hview.value, hview.variance = fill_with_interpolation_1d(hview)
                new_hist.view(flow=flow)[idx,ids_new,...] = hview
                
    return new_hist

def make_split(ratio: float, gap: float = 0., ptype: str ="step") -> Any:
    from matplotlib.gridspec import GridSpec
    cax = plt.gca()
    visible = True if ratio in [0, 1] else False
    plt.setp(cax.get_xticklabels(), visible=visible)
    plt.setp(cax.get_yticklabels(), visible=visible)

    box = cax.get_position()
    xmin, ymin = box.xmin, box.ymin
    xmax, ymax = box.xmax, box.ymax

    if ratio == 1:
        return cax, None
    elif ratio == 0:
        return None, cax
    gs = GridSpec(
        2,
        1,
        height_ratios=[ratio, 1 - ratio],
        left=xmin, right=xmax,
        bottom=ymin, top=ymax
    )
    gs.update(hspace=gap)
    ax = plt.subplot(gs[0])
    plt.setp(ax.get_xticklabels(), visible=False)
    bx = plt.subplot(gs[1], sharex=ax)

    return ax, bx

def mcplot(
    pred: List[hist.Hist] | hist.Hist | hist.Stack, # either a list of MC or a boost hist with sample axis
    data: List[hist.Hist] | hist.Hist = None, 
    syst: List[hist.Hist] | hist.Hist = None, 
    proc_axis_name: str = 'process',
    syst_axis_name: str = 'systematic',
    bin_width_norm: float | None = None,
    no_ratios: bool = False,
    combine_fit: str = 'pre-combine',
    combine_uncertainty_histo: hist.Hist | None = None,
    combine_histo_edges: Iterable[int] | Iterable[float] | None = None,
    **kwargs) -> Any:
    # inspired from boost Hist
    ax, bx = make_split(1 if no_ratios else 0.7)

    x_vals = None
    l_edge = None
    r_edge = None
    pred_values = None

    ax.set_title(kwargs.get('title', ''))
    if isinstance(pred, hist.Stack):
        pred_hstk = pred
        pred_ksum = sum(pred)
        pred_values = pred_ksum.values(0)
        x_vals = pred_ksum.axes.centers[0]
        l_edge = pred_ksum.axes.edges[0][:-1]
        r_edge = pred_ksum.axes.edges[0][1:]
    if isinstance(pred, hist.Hist):
        pred_hstk = pred
        pred_ksum = pred
        pred_values = pred.values(0)
        x_vals = pred.axes.centers[0]
        l_edge = pred.axes.edges[0][:-1]
        r_edge = pred.axes.edges[0][1:]
    if isinstance(pred, List):
        pass
    
    if "colors" in kwargs:
        ax.update({"prop_cycle":cycler(color=kwargs["colors"])})
        
    pred_hstk.plot(ax=ax, stack=True, histtype="fill", binwnorm=bin_width_norm)
    if combine_fit == 'pre-combine':
        pred_stat_error = np.sqrt(np.abs(pred_ksum.values(0)))
    else:
        if combine_uncertainty_histo is not None:
            # This is actually the TOTAL uncertainty, stat + syst from combine
            # may be named (total{_background,_signal} from fitDiagnostics
            # or Total{Procs,Bkg,Sig} if from PostFitShapesFromWorkspace
            pred_stat_error = np.sqrt(combine_uncertainty_histo.variances())
        else:
            # If we're missing the total uncertainty from a combine fit (prefit, fit_b, or fit_s) then don't draw any uncertainty band
            pred_stat_error = np.zeros_like(pred_ksum.values(0))
    if bin_width_norm is not None:
        # follow https://github.com/scikit-hep/mplhep/blob/main/src/mplhep/plot.py#L851-L858, in 1D case
        pred_stat_error = pred_stat_error * bin_width_norm / (r_edge - l_edge)
        pred_values_without_binwnorm = pred_values # needed for ratio later
        pred_values = pred_values * bin_width_norm / (r_edge - l_edge)
    ax.bar( 
        x_vals, 
        height= 2*pred_stat_error,
        width= r_edge - l_edge,
        bottom= pred_values - pred_stat_error,
        fill=False,
        linewidth=0,
        edgecolor="gray",
        hatch=4 * "/",
    )
    if bx is not None:
        bx.axhline(
            1, color="black", linestyle="dashed", linewidth=1.0
        )
    
        # MC stat error bars
        ratio = np.ones_like(pred_values)
        ratio_uncert = np.zeros_like(pred_values)

        bx.bar(
            x_vals,
            height = np.divide(2*pred_stat_error, pred_values, where=pred_values!=0),
            width  = r_edge - l_edge,
            bottom = 1 - np.divide(pred_stat_error,pred_values, where=pred_values!=0),
            color  = "red" if (combine_fit == "pre-combine") else "blue",
            alpha  = 0.4,
            label="stat" if (combine_fit == "pre-combine") else "stat+syst",
        )

    if data is not None:
        data.plot(ax=ax, color='black', histtype='errorbar', binwnorm=bin_width_norm)
        numerator_without_binwnorm = data.values(0)
        numerator = data.values(0) if not bin_width_norm else data.values(0) * bin_width_norm / (r_edge - l_edge)
        # numerator and pred_values are bin_width_normalized if active
        ratio = np.divide(numerator, pred_values, where=pred_values!=0)
        if bx is not None:
            ratio_uncert = ratio_uncertainty(
                num=numerator_without_binwnorm,
                denom=pred_values_without_binwnorm if bin_width_norm is not None else pred_values,
            )
            bx.errorbar(
                x_vals,
                ratio,
                yerr=ratio_uncert,
                color="black",
                marker="o",
                linestyle="none",
            )
        
    if syst is not None and combine_fit == "pre-combine":
        syst_list = set([
            i.replace('Up','').replace('Down','') for i in syst.axes[syst_axis_name]
        ])
        syst_up = []
        syst_dw = []
        for s in syst_list:
            if 'nominal'==s: continue
            #if 'QCD' in s: continue
            # if 'PDF' in s: continue
            #if 'JES_' in s: continue
            # if 'ElectronEn' in s: continue
            # if 'UEPS_' in s: continue
            # if 'btag_sf_stat' in s: continue
            #if 'btag_sf_light' in s: continue
            # if 'trigger' in s: continue
            # if 'Lepton' in s: continue
            
            shape_up = sum([_hs[{syst_axis_name : s + 'Up'  }] for _hs in syst]).values(0)
            shape_dw = sum([_hs[{syst_axis_name : s + 'Down'}] for _hs in syst]).values(0)
            if bin_width_norm is not None:
                shape_up = shape_up * bin_width_norm / (r_edge - l_edge)
                shape_dw = shape_dw * bin_width_norm / (r_edge - l_edge)
            
            var_up = np.where(
                np.divide(np.abs(shape_up - pred_values), pred_values, where=shape_up!=0)>10, 
                pred_values - np.abs(shape_dw - pred_values), shape_up
            )
            
            var_dw = np.where(
                np.divide(np.abs(shape_dw - pred_values), pred_values, where=shape_dw!=0)>10,
                pred_values - np.abs(shape_up - pred_values), shape_dw
            )
            
            # removing bogus normalisations
            var_up = np.where((var_up <= 0) | np.isinf(var_up) | np.isnan(var_up), pred_values, var_up)
            var_dw = np.where((var_dw <= 0) | np.isinf(var_dw) | np.isnan(var_up), pred_values, var_dw)
            
            syst_up.append((pred_values-var_up))
            syst_dw.append((pred_values-var_dw))
        
        if bx is not None:
            syst_up = np.array(syst_up)
            syst_dw = np.array(syst_dw)

            syst_uncert_up = np.sqrt(np.sum(np.power(syst_up,2), axis=0) + np.power(pred_stat_error,2))
            syst_uncert_dw = np.sqrt(np.sum(np.power(syst_dw,2), axis=0) + np.power(pred_stat_error,2))

            bx.bar(
                x_vals,
                height = np.divide(syst_uncert_up + syst_uncert_dw, pred_values, where=pred_values!=0),
                width  = r_edge - l_edge,
                bottom = np.divide(pred_values - syst_uncert_dw, pred_values, where=pred_values!=0),
                color  = "blue", alpha  = 0.2, zorder = 0,
                label="stat+syst"
            )
    
    if isinstance(pred, hist.Hist):
        if proc_axis_name in pred.axes.name:
            pred.stack(proc_axis_name).plot(
                ax=ax, stack=True, histtype="fill", binwnorm=bin_width_norm
            )
    if (data is not None) and isinstance(pred, hist.Hist):
        data.plot(ax=ax, color='black', histtype='errorbar', binwnorm=bin_width_norm)
    ax.set_xlim(l_edge[0], r_edge[-1])
    if combine_fit != "pre-combine" and combine_histo_edges is not None:
        # override to fix combine stripping the axis edges from the histograms, replacing them with bin numbers
        assert len(combine_histo_edges) == len(bx.get_xticks()), f"mismatch of edges({combine_histo_edges}) and xticks({bx.get_xticks()})"
        if bx is not None:
            bx.set_xticks(bx.get_xticks(), labels=combine_histo_edges)
    fit_label = {"pre-combine": "Prefit", "prefit": "Prefit", "fit_b": "Postfit (background only)", "fit_s": "Postfit (s+b)"}[combine_fit]
    # ax.text(
    #     0.02, 0.05, fit_label,
    #     color="black",
    #     fontsize=15, horizontalalignment='left',
    #     verticalalignment='bottom',
    #     transform=ax.transAxes
    # )
    ax.legend(ncol=2, loc='upper right', fontsize=15)
    if bx is not None:
        bx.legend(loc='upper right', fontsize=15)
        bx.set_xlabel(pred_ksum.axes[0].label)
        bx.set_ylabel('data/mc')
    ax.set_ylabel('events' if not bin_width_norm else 'events/GeV')
    
    return ax, bx

def check_systematic(
    pred: List[hist.Hist] | hist.Hist | hist.Stack, # either a list of MC or a boost hist with sample axis
    syst: List[hist.Hist] | hist.Hist | hist.Stack = None, 
    syst_axis_name: str = 'systematic',
    plot_file_name: str = 'check-sys',
    output_dir: str = './systematic-check/',
    xrange: List = [],
    bin_width_norm: float | None = None,
    no_ratios: bool = False,
    **kwargs) -> Any:
    # inspired from boost Hist
    if bin_width_norm is not None:
        raise NotImplementedError("bin width normalisation not implemented for systematic checks")
    
    if not os.path.isdir(os.path.dirname(output_dir)):
        os.mkdir(os.path.dirname(output_dir))

    if isinstance(pred, hist.Stack) or isinstance(pred, List):
        pred = sum(pred)
        
    pred_values = pred.values(0)   
    x_vals = pred.axes.centers[0]
    l_edge = pred.axes.edges[0][:-1]
    r_edge = pred.axes.edges[0][1:]
    
    pred_stat_error = np.sqrt(pred.values(0))
   
    if syst is not None:
        syst_list = set([
            i.replace('Up','').replace('Down','') for i in syst.axes[syst_axis_name]
        ])
        for s in syst_list:
            if 'nominal'==s: continue
            # if 'PDF'==s: continue
            syst_uncert_up = sum([_hs[{syst_axis_name : s + 'Up'  }] for _hs in syst]).values(0)
            syst_uncert_dw = sum([_hs[{syst_axis_name : s + 'Down'}] for _hs in syst]).values(0)
            syst_uncert_up[np.isnan(syst_uncert_up)] = 0
            syst_uncert_dw[np.isnan(syst_uncert_dw)] = 0
            
            # drawing the plots
            fig = plt.figure(figsize=(6,7))
            ax, bx = make_split(1 if no_ratios else 0.7)
            
            ax.set_title(f'{plot_file_name} : {s}')
            pred.plot(ax=ax, color='black', histtype='step', label='nominal')
            ax.hist(
                x_vals, bins=pred.axes[0].edges,
                weights= syst_uncert_up, lw=1.5,
                color='red', histtype='step', 
                label='Up'
            )
            ax.hist(
                x_vals, bins=pred.axes[0].edges,
                weights= syst_uncert_dw, lw=1.5,
                color='blue', histtype='step', 
                label='Down'
            )
            
            ax.bar( 
                x_vals, 
                height= 2*pred_stat_error,
                width=r_edge - l_edge,
                bottom= pred.values(0) - pred_stat_error,
                fill=False,
                linewidth=0,
                edgecolor="gray",
                hatch=4 * "/",
            )
            ax.set_xlim(l_edge[0], r_edge[-1])
            ax.legend(ncol=2, loc='upper right')
            ax.set_ylabel('events')
            ax.set_yscale('log')

            if bx is not None:
                bx.axhline(
                    1, color="black", linestyle="dashed", linewidth=1.0
                )
                bx.bar(
                    x_vals,
                    height = np.divide(2*pred_stat_error, pred_values, where=pred_values!=0),
                    width  = r_edge - l_edge,
                    bottom = np.divide(pred_values - pred_stat_error, pred_values, where=pred_values!=0),
                    color  = "grey",
                    alpha  = 0.4,
                )

                bx.hist(
                    x_vals, bins=pred.axes[0].edges,
                    weights=np.divide(syst_uncert_up, pred_values, where=pred_values!=0),
                    lw=1.5,
                    color='red', histtype='step',
                    label='Up'
                )
                bx.hist(
                    x_vals, bins=pred.axes[0].edges,
                    weights= np.divide(syst_uncert_dw, pred_values, where=pred_values!=0),
                    lw=1.5,
                    color='blue', histtype='step',
                    label='Down'
                )
                bx.set_ylim([0.4,1.6])
                bx.set_xlabel(pred.axes[0].label)
                bx.set_ylabel('data/mc')
            
                if len(xrange) > 0:
                    bx.set_xlim(xrange)
            
            fig.savefig(f'{output_dir}/{plot_file_name}-{pred.axes[0].name}-{s}.pdf')
            fig.savefig(f'{output_dir}/{plot_file_name}-{pred.axes[0].name}-{s}.png')

def plotting(config, variable, channel, rebin=1, xlim=[], blind=False, era="someyear", checksyst=True,
             remap_replacement_types = None, bin_width_norm=None, no_ratios=False,
             combine_fit="pre-combine", combine_total_uncertainty="total_background", combine_channel_group=None) -> None:
    assert combine_fit in ["pre-combine", "prefit", "fit_b", "fit_s"]
    if remap_replacement_types is None:
        remap_replacement_types = [] #expected args: "datadriven", "validation"
    datasets:Dict = dict()
    color_cycle:List = []
    edges:Iterable[str] | Iterable[float] | None = None
    combine_uncertainty_histo: hist.Hist | None = None
    combine_channels: List[str] | None = None
    combine_lumi: int | None = None
    combine_era: str | None = None

    if bin_width_norm is None and "bin_width_norm" in config:
        bin_width_norm = config.bin_width_norm

    for ng, name in enumerate(config.groups):
        if combine_fit == "pre-combine":
            # handle the plotting of histograms directly from SMQawa
            histograms = dict(
                filter(
                    lambda _n: _n[0] in config.groups[name].processes,
                    config.boosthist.items()
                )
            )
            p = datagroup(
                histograms       = histograms,
                ptype            = config.groups[name].type,
                observable       = variable,
                name             = name,
                xsections        = config.xsections,
                channel          = channel,
                luminosity       = config.luminosity.value,
                rebin            = rebin,
                remap_class_name = config.groups[name].remap_class_name if "remap_class_name" in config.groups[name] else None,
            )
            #remap_replacement_types lets us control whether we replace a given process with a remap type, such as a datadriven estimate. 
            # The remap_class should have a method which returns a tuple of the config group name for which a remapped group replaces, and what type it is categorized as
            # for example, in WZ, we have a data driven estimate for SR0 and SR1 derived from B0 and B1, and these are called "datadriven" to indicate they are for full replacement
            # of the DY MonteCarlo
            # Meanwhile, we can do some crossvalidation/closure tests by looking at the datadriven etimate derived for other regions, so their type is "validation"
            # to toggle datadriven types and/or validation types (or any other type name you choose) to replace the given process, just add it to the remap_replacement_types list
            if p.remap_replace_group_name is not None:
                if p.remap_replace_type in remap_replacement_types:
                    print(f"Overwriting: channel: {p.channel} type: {p.remap_replace_type}, {p.remap_replace_group_name} replaced by {p.name}")
                    # overwrite a previously defined dataset in the dictionary. This requires the remap types to be after ALL MC in the config file (and still before the real data)
                    datasets[p.remap_replace_group_name] = p
                    if hasattr(config.groups[name], "color") and len(p.to_boost().shape):
                        # must replace the previous color cycler...
                        index = list(datasets.keys()).index(p.remap_replace_group_name)
                        color_cycle[index] = config.groups[name].color
                else:
                    print(f"Skipping: channel: {p.channel} type: {p.remap_replace_type}, {p.remap_replace_group_name} would have been replaced by {p.name}")
                    # this process is ignored / not added to the stack
                    continue
            else:
                # nominal path for MC/data which doesn't have a remap_class and 
                datasets[p.name] = p
                # add the new color to the color cycler...
                if hasattr(config.groups[name], "color") and len(p.to_boost().shape):
                    color_cycle.append(config.groups[name].color)
            if p.ptype == "signal":
                signal = p.name
        else:
            # handle the combine prefit or postfit inputs similarly to the pre-combine path, with a channel selection ala dctools.datagroup
            if ng == 0:
                if combine_channel_group is not None:
                    combine_channels_config = config.combinehist[name]["channel_groups"][variable][combine_channel_group]
                    combine_channels = combine_channels_config.channels
                    combine_lumi = combine_channels_config.luminosity
                    combine_era = combine_channels_config.era
                else:
                    combine_channels = channel
                    combine_era = era
                combine_uncertainty_histo = config.combinehist[combine_total_uncertainty][combine_fit][variable][{"channel": combine_channels}]
                edges = config.combinehist[name]["edges"][variable]
            p = config.combinehist[name][combine_fit][variable][{"channel": combine_channels}]
            # for the 'systematic' axis and with adding/regularizing the '
            datasets[name] = p
            if config.groups[name].type == "signal":
                signal = name
            if rebin != 1:
                raise NotImplementedError("for combine prefit/fit_b/fit_s plotting the rebin functionality has not been implemented")

            if hasattr(config.groups[name], "color") and (hasattr(p, "shape") and len(p.shape)) or len(p.to_boost().shape):
                color_cycle.append(config.groups[name].color)
    
    if combine_fit == "pre-combine":
        _plot_channel = plotter.add_process_axis(datasets)
    else:
        _plot_channel = dctools.dict_to_hist_axis(datasets, axis_name='process', axis_label=None, axis_type = 'StrCategory')
        combine_uncertainty_histo = combine_uncertainty_histo.project('systematic', variable)
    # projection must avoid the variable rebinning bug, this is a workaround and can be replaced by just variable once fixed: https://github.com/scikit-hep/hist/issues/639
    variable_in_axes = variable if variable in _plot_channel.axes.name else ""
    pred = _plot_channel.project('process', 'systematic', variable_in_axes)[:hist.loc('data'),:,:]
    data = _plot_channel[{'systematic':'nominal'}].project('process', variable_in_axes)[hist.loc('data'),:]
    

    plt.figure(figsize=(6, 4.9 if no_ratios else 7))
    ax, bx = plotter.mcplot(
        pred[{'systematic':'nominal'}].stack('process'),
        data=None if blind else data,
        syst=pred.stack('process'),
        colors = color_cycle,
        no_ratios=no_ratios,
        combine_fit=combine_fit,
        combine_uncertainty_histo=combine_uncertainty_histo[{'systematic':'nominal'}] if combine_uncertainty_histo else None,
        combine_histo_edges=edges,
    )

    ymax = np.max([10000]+[c.get_height() for c in ax.containers[0] if ~np.isnan(c.get_height())])
    ymin = np.min([0.001]+[c.get_height() for c in ax.containers[0] if ~np.isnan(c.get_height())])

    ax.set_ylim(0.001, 1000*ymax)
    try:
        sig_ewk = _plot_channel[{'systematic':'nominal'}].project('process', variable)[hist.loc('VBSZZ2l2nu'),:]   
        sig_qcd = _plot_channel[{'systematic':'nominal'}].project('process', variable)[hist.loc('ZZ2l2nu'),:]   
        sig_ewk.plot(ax=ax, histtype='step', color='red', binwnorm=bin_width_norm)
        sig_qcd.plot(ax=ax, histtype='step', color='purple', binwnorm=bin_width_norm)
    except:
        pass
    if bx is not None:
        bx.set_ylim([0.1, 1.9])
        if len(xlim) > 0:
            bx.set_xlim(xlim)
    elif len(xlim) > 0:
        ax.set_xlim(xlim)
    ax.set_title(f"channel {combine_channel_group or channel}: {combine_era or era}")
    hep.cms.label("", ax=ax, data=not blind, lumi=combine_lumi, year=combine_era or int(era)) #add lumi=lumi, add year=int(era) with handling of APV, etc.
    ax.set_yscale('log')

    cmb_postfix = "-" + combine_fit if combine_fit in ["prefit", "fit_b", "fit_s"] else ""
    rrt_postfix = "-" + "-".join(remap_replacement_types) if (isinstance(remap_replacement_types, list) and len(remap_replacement_types) > 0 and not (len(remap_replacement_types) == 1 and remap_replacement_types[0] == "nothing")) else ""
    nrat_postfix = "-noratio" if no_ratios else ""
    gbwn_postfix = f"-binwnorm{bin_width_norm}".replace(".", "p") if bin_width_norm is not None else ""
    print(gbwn_postfix)
    #xlim_postfix = f"-xlim{int(xlim[0])}-{int(xlim[1])}" if len(xlim) == 2 else ""
    plt.savefig(f'plot-{combine_channel_group or channel}-{variable}-{combine_era or era}{cmb_postfix}{rrt_postfix}{nrat_postfix}{gbwn_postfix}.pdf')
    plt.savefig(f'plot-{combine_channel_group or channel}-{variable}-{combine_era or era}{cmb_postfix}{rrt_postfix}{nrat_postfix}{gbwn_postfix}.png')
    plt.clf()
    
    if checksyst:
        pred = _plot_channel.project('process','systematic', variable)[:hist.loc('data'),:,:]
        data = _plot_channel[{'systematic':'nominal'}].project('process',variable)[hist.loc('data'),:] 
        plotter.check_systematic(
            pred[{'systematic':'nominal'}].stack('process'),
            syst=pred.stack('process'),
            plot_file_name=f'check-sys-{channel}-{era}', 
            xrange=xlim,
            no_ratios=no_ratios,
            bin_width_norm=bin_width_norm,
        )
        plt.clf()
    return _plot_channel, datasets
