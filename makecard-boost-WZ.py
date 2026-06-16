import yaml
import os
import json
import gzip
import pickle
import argparse
import dctools
import hist
import matplotlib.pyplot as plt
from dctools import plot as plotter
from typing import Any, IO
import numpy as np
import rich
import warnings
# statsmodels is imported lazily inside smooth_diff_lowess() so it is only required when LOWESS smoothing is actually requested.

class config_input:
    def __init__(self, cfg):
        self._cfg = cfg 

    def __getitem__(self, key):
        v = self._cfg[key]
        if isinstance(v, dict):
            return config_input(v)

    def __getattr__(self, k):
        try:
            v = self._cfg[k]
            if isinstance(v, dict):
                return config_input(v)
            return v
        except:
            return None

    def __iter__(self):
        return iter(self._cfg)


class config_loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""
    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""
        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir
        super().__init__(stream)


def construct_include(loader: config_loader, node: yaml.Node) -> Any:
    """Include file referenced at node."""
    filename = os.path.abspath(os.path.join(loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, config_loader)
        elif extension in ('json', ):
            return json.load(f)
        else:
            return ''.join(f.readlines())

yaml.add_constructor('!include', construct_include, config_loader)

# def smooth_ratio_lowess(nominal, variation, frac=0.4):
#     if isinstance(variation, tuple):
#         return tuple(smooth_ratio_lowess(nominal, var, frac=frac) for var in variation)
#     nom = nominal.view(flow=False)['value']
#     var = variation.view(flow=False)['value']

#     if nom.ndim != 1:
#         raise ValueError("smooth_ratio_lowess only supports 1D histograms.")

#     mask = nom > 0
#     ratio = np.ones_like(nom)
#     ratio[mask] = var[mask] / nom[mask]

#     x = np.arange(len(ratio))
#     ratio_smooth = lowess(ratio, x, frac=frac, return_sorted=False)
#     # diff = var - nom
#     # x = np.arrange(len(diff))
#     # diff_smooth = lowess(diff, x, frac=frac, return_sorted=False)

#     var_smooth = ratio_smooth * nom
#     # var_smooth = diff_smooth + nom
#     var_smooth[var_smooth < 0] = 0.0

   
#     new_hist = variation.copy()
#     new_hist.view(flow=False)['value'][:] = var_smooth

#     return new_hist

def smooth_diff_lowess(nominal, variation, frac=0.4):
    from statsmodels.nonparametric.smoothers_lowess import lowess
    if isinstance(variation, tuple):
        return tuple(smooth_diff_lowess(nominal, var, frac=frac) for var in variation)
    nom = nominal.view(flow=False)['value']
    var = variation.view(flow=False)['value']

    if nom.ndim != 1:
        raise ValueError("smooth_ratio_lowess only supports 1D histograms.")

    mask = nom > 0
    diff = var - nom
    x = np.arange(len(diff))
    diff_smooth = np.zeros_like(diff)
    diff_smooth[mask] = lowess(diff[mask], x[mask], frac=frac, return_sorted=False)
    var_smooth = diff_smooth + nom
    var_smooth = np.clip(var_smooth, 0, None)

    new_hist = variation.copy()
    new_hist.view(flow=False)['value'][:] = var_smooth

    return new_hist

def main():
    parser = argparse.ArgumentParser(description='The Creator of Combinators')
    parser.add_argument("-i"  , "--input"   , type=str , default="./config/input_UL_2018-WZ_inclusive.yaml")
    parser.add_argument("-v"  , "--variable", type=str , default="dilep_pt")
    parser.add_argument("-y"  , "--era"     , type=str , default='2018')
    parser.add_argument("-c"  , "--channel" , nargs='+', type=str)
    parser.add_argument("-s"  , "--signal"  , nargs='+', type=str)
    parser.add_argument('-n'  , "--name"    , type=str , default='')
    parser.add_argument('-p'  , "--plot"    , action="store_true")
    parser.add_argument('--rebin', type=int, nargs='+', default=None, help='rebin card by integer value or list of bin groups')
    parser.add_argument("--bins",
            type=lambda s: [float(item) for item in s.split(',')],
            help='input a comma separated list. ex: --bins="-1.2,0,1.2"'
    )
    parser.add_argument('--blind', action='store_true', help='blinding the channel')
    parser.add_argument('--checksyst', action='store_true')
    parser.add_argument("-rrt", "--remap_replacement_types", nargs='*', type=str, default=[])

    options = parser.parse_args()
    config = dctools.read_config(options.input)

    print(f'making: {options.channel} : {options.variable} : {options.era}')

    if len(options.channel) == 1:
        options.channel = options.channel[0]

    # make datasets per prcess
    datasets = {}
    signal = ""

    if options.name=='':
        options.name == options.channel


    datasets:Dict = dict()
    for name in config.groups:
        histograms = dict(
            filter(
                lambda _n: _n[0] in config.groups[name].processes,
                config.boosthist.items()
            )
        )
        v_cfg = config.plotting[options.channel][options.variable]
        if options.rebin is not None:
            rebin = options.rebin
        elif "rebin" in v_cfg:
            rebin = v_cfg.rebin
        else:
            rebin = 1
        if isinstance(rebin, list) and len(rebin) < 2:
            rebin = rebin[0]

        p = dctools.datagroup(
            histograms = histograms,
            ptype      = config.groups[name].type,
            observable = options.variable,
            name       = name,
            xsections  = config.xsections,
            channel    = options.channel,
            luminosity = config.luminosity.value,
            rebin      = rebin,
            remap_class_name = config.groups[name].remap_class_name if "remap_class_name" in config.groups[name] else None,
            # era        = options.era,
        )

        #remap_replacement_types lets us control whether we replace a given process with a remap type, such as a datadriven estimate.
        # The remap_class should have a method which returns a tuple of the config group name for which a remapped group replaces, and what type it is categorized as
        # for example, in WZ, we have a data driven estimate for SR0 and SR1 derived from B0 and B1, and these are called "datadriven" to indicate they are for full replacement
        # of the DY MonteCarlo
        # Meanwhile, we can do some crossvalidation/closure tests by looking at the datadriven etimate derived for other regions, so their type is "validation"
        # to toggle datadriven types and/or validation types (or any other type name you choose) to replace the given process, just add it to the remap_replacement_types list
        if p.remap_replace_group_name is not None:
            if p.remap_replace_type in options.remap_replacement_types:
                rich.print(f"Overwriting: channel: [red]{p.channel}[/red] type: {p.remap_replace_type}, [yellow]{p.remap_replace_group_name}[/yellow] replaced by [green]{p.name}")
                # overwrite a previously defined dataset in the dictionary. This requires the remap types to be after ALL MC in the config file (and still before the real data)
                p.remap_original_group_name = p.name
                p.name = p.remap_replace_group_name
                datasets[p.remap_replace_group_name] = p
            else:
                print(f"Skipping: channel: {p.channel} type: {p.remap_replace_type}, {p.remap_replace_group_name} would have been replaced by {p.name}")
                # this process is ignored / not added to the stack
                continue
        else:
            # nominal path for MC/data which doesn't have a remap_class and
            datasets[p.name] = p
        if p.ptype == "signal":
            signal = p.name


    if options.plot:
        _plot_channel = plotter.add_process_axis(datasets)
        pred = _plot_channel.project('process','systematic', options.variable)[:hist.loc('data'),:,:]
        data = _plot_channel[{'systematic':'nominal'}].project('process',options.variable)[hist.loc('data'),:] 

        plt.figure(figsize=(6,7))
        ax, bx = plotter.mcplot(
            pred[{'systematic':'nominal'}].stack('process'),
            data=None if options.blind else data,
            syst=pred.stack('process'),
        )

        try:
            # sig_ewk = _plot_channel[{'systematic':'nominal'}].project('process', variable)[hist.loc('WZ_ewk'),:]
            sig_qcd = _plot_channel[{'systematic':'nominal'}].project('process', variable)[hist.loc('WZ'),:]
            # sig_ewk.plot(ax=ax, histtype='step', color='red')
            sig_qcd.plot(ax=ax, histtype='step', color='purple')
        except:
            pass

        ymax = np.max([line.get_ydata().max() for line in ax.lines if line.get_ydata().shape[0]>0])
        ymin = np.min([line.get_ydata().min() for line in ax.lines if line.get_ydata().shape[0]>0])

        ax.set_ylim(0.001, 100*ymax)
        ax.set_title(f"channel {options.channel}: {options.era}")

        ax.set_yscale('log')
        plt.savefig(f'plot-{options.channel}-{options.variable}-{options.era}.pdf')


    if options.checksyst:
        _plot_channel = plotter.add_process_axis(datasets)
        pred = _plot_channel.project('process','systematic', options.variable)[:hist.loc('data'),:,:]
        data = _plot_channel[{'systematic':'nominal'}].project('process',options.variable)[hist.loc('data'),:]
        plotter.check_systematic(
            pred[{'systematic':'nominal'}].stack('process'),
            syst=pred.stack('process'),
            plot_file_name=f'check-sys-{options.channel}-{options.era}'
        )

    card_name = options.channel+options.era

    card = dctools.datacard(
        name = signal if len(options.name)==0 else options.name,
        channel= card_name
    )
    card.shapes_headers()

    data_obs = datasets.get("data").get("nominal")
    
    card.add_observation(data_obs)

    rich.print("[yellow]btag uncertainties disabled")
    for _, p in datasets.items():
        # Systematics Conventions: https://gitlab.cern.ch/cms-analysis/general/systematics/-/blob/master/systematics_master.yml?ref_type=heads
        print(" --> ", p.name)
        if len(p.to_boost().shape) == 0 or p.get("nominal").sum().value == 0:
            print(f"--> histogram for the process {p.name} is empty !")
            continue

        if p.ptype=="data":
            continue
        if not card.add_nominal(p.name, p.get("nominal"), p.ptype): continue
        year = options.era.replace('APV','')
        
        # luminosity
        if options.era in ["2016", "2016APV", "2017", "2018"]:
            try:
                card.add_log_normal(p.name, f"CMS_lumi_{options.era}", config.luminosity.uncer)
                warnings.warn("Old style single-lumi uncertainty detected, please update to uncer_<lumi_uncertainty>"
                              "\n where <lumi_uncertainty> is one of lumi_13TeV_1516_l, lumi_13TeV_151617_l, lumi_13TeV_15161718_l"
                              "\n See https://twiki.cern.ch/twiki/bin/viewauth/CMS/LumiRecommendationsRun2 for more details")
            except Exception as e:
                pass
            try:
                card.add_log_normal(p.name, f"CMS_lumi_{options.era}", getattr(config.luminosity, f"uncer_lumi_{options.era}"))
                card.add_log_normal(p.name, f"CMS_lumi_13TeV_1718", getattr(config.luminosity, f"uncer_lumi_13TeV_1718"))
                card.add_log_normal(p.name, f"CMS_lumi_13TeV_correlated", getattr(config.luminosity, f"uncer_lumi_13TeV_correlated"))
                warnings.warn("Old style lumi uncertainty detected, please update to uncer_<lumi_uncertainty>"
                              "\n where <lumi_uncertainty> is one of lumi_13TeV_1516_l, lumi_13TeV_151617_l, lumi_13TeV_15161718_l"
                              "\n See https://twiki.cern.ch/twiki/bin/viewauth/CMS/LumiRecommendationsRun2 for more details")
            except Exception as e:
                pass
            try:
                lumi1516 = getattr(config.luminosity, f"uncer_lumi_13TeV_1516_l")
                lumi151617 = getattr(config.luminosity, f"uncer_lumi_13TeV_151617_l")
                lumi15161718 = getattr(config.luminosity, f"uncer_lumi_13TeV_15161718_l")
                if lumi1516 is not None:
                    card.add_log_normal(p.name, f"CMS_lumi_13TeV_1516_l", lumi1516)
                if lumi151617 is not None:
                    card.add_log_normal(p.name, f"CMS_lumi_13TeV_151617_l", lumi151617)
                if lumi15161718 is not None:
                    card.add_log_normal(p.name, f"CMS_lumi_13TeV_15161718_l", lumi15161718)
            except Exception as e:
                pass
        else:
            raise NotImplementedError("non-Run2 Luminosity uncertainties not implemented yet")

        if p.remap_replace_group_name is not None:
            # If we later decide to add shape nuisances to e.g. datadriven estimates, we'll need to eliminate or alter this code path
            rich.print(f"[red]Skipping shape nuisances for [green]{p.name} (remap_original_group_name={p.remap_original_group_name})")
        else:
            # scale factors / resolution
            nominal = p.get("nominal")
            card.add_shape_nuisance(p.name, f"CMS_res_e_{options.era}"  , p.get("ElectronEn"), symmetrise=False)
            # res_e = p.get("ElectronEn")
            # if res_e is not None:
            #     res_e_smooth = smooth_diff_lowess(nominal, res_e, frac=0.3)
            #     card.add_shape_nuisance(p.name, f"CMS_res_e_{options.era}", res_e_smooth, symmetrise=False)

            # res_m = p.get("MuonRoc")
            # if res_m is not None:
            #     res_m_smooth = smooth_diff_lowess(nominal, res_m, frac=0.4)
            #     card.add_shape_nuisance(p.name, f"CMS_res_m_{options.era}", res_m_smooth, symmetrise=False)

            # res_t = p.get("TauEn")
            # if res_t is not None:
            #     res_t_smooth = smooth_diff_lowess(nominal, res_t, frac=0.4)
            #     card.add_shape_nuisance(p.name, f"CMS_res_t_{options.era}", res_t_smooth, symmetrise=False)



            card.add_shape_nuisance(p.name, f"CMS_res_m_{options.era}"  , p.get("MuonRoc")   , symmetrise=False)
            card.add_shape_nuisance(p.name, f"CMS_res_t_{options.era}"  , p.get("TauEn")   , symmetrise=False)
            card.add_shape_nuisance(p.name, f"CMS_lept_sf_{options.era}", p.get("LeptonSF")  , symmetrise=False)
            card.add_shape_nuisance(p.name, f"CMS_trig_sf_{options.era}", p.get("triggerSF") , symmetrise=False)

            # JES/JES and UEPS
            # card.add_shape_nuisance(p.name, f"CMS_jes_{options.era}", p.get("JES"), symmetrise=False)

            card.add_shape_nuisance(p.name, f"JES_Absolute{year}"      , p.get(f"JES_Absolute{year}")      , symmetrise=False)
            card.add_shape_nuisance(p.name, f"JES_BBEC1{year}"         , p.get(f"JES_BBEC1{year}")         , symmetrise=False)
            card.add_shape_nuisance(p.name, f"JES_EC2{year}"           , p.get(f"JES_EC2{year}")           , symmetrise=False)
            card.add_shape_nuisance(p.name, f"JES_HF{year}"            , p.get(f"JES_HF{year}")            , symmetrise=False)
            card.add_shape_nuisance(p.name, f"JES_RelativeSample{year}", p.get(f"JES_RelativeSample{year}"), symmetrise=False)

            card.add_shape_nuisance(p.name, f"JES_Absolute"   , p.get("JES_Absolute")   , symmetrise=False)
            card.add_shape_nuisance(p.name, f"JES_BBEC1"      , p.get("JES_BBEC1")      , symmetrise=False)
            card.add_shape_nuisance(p.name, f"JES_EC2"        , p.get("JES_EC2")        , symmetrise=False)
            card.add_shape_nuisance(p.name, f"JES_HF"         , p.get("JES_HF")         , symmetrise=False)
            card.add_shape_nuisance(p.name, f"JES_RelativeBal", p.get("JES_RelativeBal"), symmetrise=False)
            card.add_shape_nuisance(p.name, f"JES_FlavorQCD"  , p.get("JES_FlavorQCD")  , symmetrise=False)

            card.add_shape_nuisance(p.name, f"CMS_jer_{options.era}", p.get("JER"), symmetrise=False)
            card.add_shape_nuisance(p.name, f"CMS_UES_{options.era}", p.get("UES"), symmetrise=False)

            #lowess smothening method
            # jer = p.get("JER")
            # if jer is not None:
            #     jer_smooth = smooth_diff_lowess(nominal, jer, frac=0.4)
            #     card.add_shape_nuisance(p.name, f"CMS_jer_{options.era}", jer_smooth, symmetrise=False)
            
            # UES = p.get("UES")
            # if UES is not None:
            #     UES_smooth = smooth_diff_lowess(nominal, UES, frac=0.3)
            #     card.add_shape_nuisance(p.name, f"CMS_UES_{options.era}", UES_smooth, symmetrise=False)

            # Can maybe ne correlated over era's?
            card.add_shape_nuisance(p.name, f"PS_FSR_{options.era}", p.get("UEPS_FSR"), symmetrise=False)
            card.add_shape_nuisance(p.name, f"PS_ISR_{options.era}", p.get("UEPS_ISR"), symmetrise=False)

            # b-tagging uncertainties
            # btag_sf_bc_2016APV, btag_sf_light_2016APV
            # try:
            #     card.add_shape_nuisance(p.name, f"CMS_btag_sf_uds_{options.era}" , p.get(f"btag_sf_light_{options.era}"), symmetrise=True)
            #     card.add_shape_nuisance(p.name, f"CMS_btag_sf_bc_{options.era}"  , p.get(f"btag_sf_bc_{options.era}")   , symmetrise=False)
            #     card.add_shape_nuisance(p.name, f"CMS_btag_df_stat_{options.era}", p.get("btag_sf_stat")            , symmetrise=False)
            # except:
            #     pass
            # # b-tagging uncertainties correlated over years
            # card.add_shape_nuisance(p.name, "CMS_btag_sf_bc"  , p.get("btag_sf_bc_correlated")   , symmetrise=False)
            # card.add_shape_nuisance(p.name, "CMS_btag_sf_uds" , p.get("btag_sf_light_correlated"), symmetrise=True)

            # other uncertainties
            card.add_shape_nuisance(p.name, f"CMS_pileup_{options.era}", p.get("pileup_weight"), symmetrise=False)


            #QCD scale, PDF and other theory uncertainty
            if 'gg' not in p.name:
                card.add_qcd_scales(
                        p.name, f"CMS_QCDScale{p.name}",
                        [p.get("QCDScale0w"), p.get("QCDScale1w"), p.get("QCDScale2w")]
            )

            # #Individual QCD scales

            # if 'gg' not in p.name:
            #     card.add_qcd_scales(p.name, f"CMS_QCDScale0{p.name}", [p.get("QCDScale0w")])
            #     card.add_qcd_scales(p.name, f"CMS_QCDScale1{p.name}", [p.get("QCDScale1w")])
            #     card.add_qcd_scales(p.name, f"CMS_QCDScale2{p.name}", [p.get("QCDScale2w")])

            # PDF uncertaintites / not working for the moment
            card.add_shape_nuisance(p.name, "pdf"   , p.get("PDF_weight"), symmetrise=False)
            card.add_shape_nuisance(p.name, "alphaS", p.get("aS_weight" ), symmetrise=False)

            # Electroweak Corrections uncertainties
            if ('WZ' in p.name):
                card.add_shape_nuisance(p.name, "ewk_corr_WZ", p.get("kEW"), symmetrise=False)
            if ('ZZ' in p.name) and ('EWK' not in p.name):
                card.add_shape_nuisance(p.name, "ewk_corr_ZZ", p.get("kEW"), symmetrise=False)

            # Add Barlow-Beeston Lite MC Stat uncertainty
            card.add_auto_stat()

        # define rates
        # define rate for DY category
        if p.name in ["DY"]:
            if "DY" in card_name:
                card.add_rate_param(f"NormDY_{options.era}", "inc-DY*", p.name)
            elif "SR" in card_name:
                card.add_rate_param(f"NormDY_{options.era}", card_name+'*', p.name)
        # elif p.name  in ["WW"]:
        #     if "inc-EM" in card_name:
        #         card.add_rate_param(f"NormWW_{options.era}", "inc-EM*", p.name)
        #     elif "SR" in card_name:
        #         card.add_rate_param(f"NormWW_{options.era}", card_name+'*', p.name)


    # saving the datacard
    card.dump()

if __name__ == "__main__":
    main()
