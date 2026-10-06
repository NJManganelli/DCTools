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
import copy
from typing import Any, IO, Dict
import numpy as np
import rich
import warnings
from statsmodels.nonparametric.smoothers_lowess import lowess


# boson -> fit observable; must match BOSON_CONFIG in SMQawa/polarization_split.py
BOSON_OBS = {
    "W":     "qcos_theta_w_reco",    # W, inclusive WZ (charge-folded angle)
    "Wp":    "cos_theta_wp_reco",    # W+
    "Wm":    "cos_theta_wm_reco",    # W-
    "Z":     "cos_theta_z_reco",     # Z, inclusive WZ
    "Z_WpZ": "cos_theta_wpz_reco",   # Z in W+Z
    "Z_WmZ": "cos_theta_wmz_reco",   # Z in W-Z
}
POLS = ("long", "left", "right")
# the 18 split-signal config groups WZ_<boson>_<pol> (only used in polarization mode)
POL_SPLIT_GROUPS = {f"WZ_{b}_{p}" for b in BOSON_OBS for p in POLS}
# WZ events with no valid gen W->tau nu + Z->ll are in none of the templates
# (~5% of the SR01 WZ yield in Run 2, ~2% in 2024). In polarization mode they are
# added back as this background, built on the fly from the inclusive WZ group.
POL_UNMATCHED_GROUPS = {f"WZ_{b}_unmatched" for b in BOSON_OBS}
# every card process that is a piece of the one WZ sample (shares its QCD-scale nuisance)
WZ_POL_PIECES = POL_SPLIT_GROUPS | POL_UNMATCHED_GROUPS
# boson -> weight prefix of its pure-state templates <obs>_<wpref>_<pol> in the
# inclusive WZ histograms (qawa.polarization naming)
BOSON_WPREF = {"W": "wZW", "Wp": "wZWp", "Wm": "wZWm",
               "Z": "wWZ", "Z_WpZ": "wWpZ", "Z_WmZ": "wWmZ"}


# ---------------------------------------------------------------------------
# Unmatched WZ: U = nominal - sum_p f_p T_p
# ---------------------------------------------------------------------------
def _hist_values(h):
    """(value, variance) arrays including flow bins; variance None for non-Weight storage."""
    v = h.view(flow=True)
    return (np.asarray(v["value"]), np.asarray(v["variance"])) if v.dtype.names else (np.asarray(v), None)


def _aligned_values(h_src, h_ref):
    """Values of h_src laid out on h_ref's channel/systematic categories (the
    StrCategory order can differ between histograms). Missing categories are zero."""
    names = [a.name for a in h_ref.axes]
    if [a.name for a in h_src.axes] != names or names[:2] != ["channel", "systematic"]:
        raise ValueError(f"[pol] axis mismatch: {[a.name for a in h_src.axes]} vs {names}")
    ref_val, _ = _hist_values(h_ref)
    src_val, _ = _hist_values(h_src)
    out = np.zeros_like(ref_val)
    src_ch, src_sy = list(h_src.axes["channel"]), list(h_src.axes["systematic"])
    missing = []
    for i, c in enumerate(h_ref.axes["channel"]):
        if c not in src_ch:
            missing.append(c)
            continue
        ic = src_ch.index(c)
        for j, s in enumerate(h_ref.axes["systematic"]):
            if s not in src_sy:
                missing.append(f"{c}/{s}")
                continue
            out[i, j] = src_val[ic, src_sy.index(s)]
    return out, missing


def pol_gen_fractions(hists, boson):
    """Generator fractions (f0, fL, fR) the processor built the templates with,
    recovered from the gen-level histograms: nominal_gen = sum_p f_p T_p^gen
    (all channels, nominal systematic), solved by least squares. Exact up to
    float precision when the pure-state weights are consistent."""
    obs, wpref = BOSON_OBS[boson], BOSON_WPREF[boson]
    gobs = obs.replace("_reco", "_gen")
    keys = [gobs] + [f"{gobs}_{wpref}_{p}" for p in POLS]
    missing = [k for k in keys if k not in hists]
    if missing:
        raise KeyError(f"[pol] gen-level histograms for {boson} not found: {missing}")

    def _flat(name):
        return hists[name][{"systematic": "nominal"}].project(gobs).values()

    nom = _flat(gobs)
    T = np.vstack([_flat(k) for k in keys[1:]]).T
    f, *_ = np.linalg.lstsq(T, nom, rcond=None)
    resid = np.max(np.abs(T @ f - nom)) / max(nom.max(), 1e-12)
    if abs(f.sum() - 1) > 1e-3 or resid > 1e-3:
        warnings.warn(f"[pol] {boson}: gen templates do not close exactly "
                      f"(sum f = {f.sum():.5f}, max rel. residual = {resid:.1e})")
    return dict(zip(("f0", "fL", "fR"), map(float, f)))


def pol_unmatched_hist(hists, boson):
    """U = nominal - sum_p f_p T_p on the nominal's axes, for every channel and
    systematic: the WZ events in none of the pure-state templates (no valid gen
    W -> tau nu + Z -> ll). Negative bins (negative MC weights at low statistics)
    are clipped to zero. The variance cannot be recovered from the inputs; it is
    the nominal variance scaled by the unmatched share of the bin."""
    obs, wpref = BOSON_OBS[boson], BOSON_WPREF[boson]
    f = pol_gen_fractions(hists, boson)
    h_nom = hists[obs]
    nval, nvar = _hist_values(h_nom)
    matched = np.zeros_like(nval)
    missing = set()
    for p, fk in zip(POLS, ("f0", "fL", "fR")):
        val, miss = _aligned_values(hists[f"{obs}_{wpref}_{p}"], h_nom)
        matched += f[fk] * val
        missing.update(miss)
    if missing:
        warnings.warn(f"[pol] {boson}: templates lack {sorted(missing)[:5]}... -- "
                      f"treated as zero, so U = nominal there")
    u = np.clip(nval - matched, 0.0, None)
    share = np.divide(u, nval, out=np.zeros_like(u), where=nval > 0)
    h_u = copy.deepcopy(h_nom)
    view = h_u.view(flow=True)
    view["value"] = u
    view["variance"] = (nvar if nvar is not None else u) * np.clip(share, 0.0, 1.0)
    return h_u, f

# modes in which an out-of-fiducial WZ background is added
FIDUCIAL_MODES = ("fiducial", "asymmetry_fiducial")

# Run 3 (13.6 TeV) eras. Lumi uncertainties are read from the config if present;
# missing values warn and are skipped rather than aborting the card build.
RUN3_ERAS = ("2022", "2022EE", "2023", "2023BPix", "2024", "2025", "2026")


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


def smooth_diff_lowess(nominal, variation, frac=0.4):
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


def charge_tag(channel):
    """Map a charge-split channel suffix to a W charge tag (or None)."""
    if channel.endswith("+") or channel.endswith("_p"):
        return "Wp"
    if channel.endswith("-") or channel.endswith("_m"):
        return "Wm"
    return None


def base_channel(channel):
    """Strip charge / fiducial suffixes so the plotting (rebin) block resolves.
    Applied repeatedly so stacked suffixes are all removed:
        inc-SR01_p_IFV -> inc-SR01 ,  inc-SR01_m_OFV -> inc-SR01"""
    sufs = ("_IFV", "_OFV", "_p", "_m", "+", "-")
    changed = True
    while changed:
        changed = False
        for suf in sufs:
            if channel.endswith(suf):
                channel = channel[:-len(suf)]
                changed = True
    return channel


def get_rebin(config, channel, variable, cli_rebin):
    """Resolve rebin: CLI override > plotting[base_channel][variable].rebin > 1."""
    if cli_rebin is not None:
        rebin = cli_rebin
    else:
        rebin = 1
        try:
            v_cfg = config.plotting[base_channel(channel)][variable]
            if "rebin" in v_cfg:
                rebin = v_cfg.rebin
        except Exception:
            rebin = 1
    if isinstance(rebin, list) and len(rebin) < 2:
        rebin = rebin[0]
    return rebin


def _shape_nuisance_bases(p):
    """Base names of every shape nuisance present on a (remapped) datagroup:
    reads the 'systematic' axis, drops 'nominal', strips trailing Up/Down.
    For the DD estimate this is exactly what the processor filled -- the DD
    stat nuisance ('DDDY' or 'stat_<era>') plus any MC systematics propagated
    through the non-DY subtraction."""
    try:
        cats = [str(c) for c in p.to_boost().axes["systematic"]]
    except Exception as e:
        warnings.warn(f"[cards] could not read systematic axis for group "
                      f"'{getattr(p, 'name', '?')}': {e}; no DD shape nuisances added.")
        return []
    bases = set()
    for c in cats:
        if c == "nominal":
            continue
        if c.endswith("Up"):
            bases.add(c[:-2])
        elif c.endswith("Down"):
            bases.add(c[:-4])
    return sorted(bases)


def add_lumi_uncertainties(card, p, config, era):
    """Attach luminosity log-normal nuisances for `p`.

    Run 2 keeps the original (correlated + per-year) scheme. Run 3 (13.6 TeV)
    reads whatever the config provides -- a per-year component
    (luminosity.uncer_lumi_<era>) and a Run-3-correlated component
    (luminosity.uncer_lumi_13p6TeV_correlated) -- and, if neither is present,
    emits a warning and builds the card WITHOUT a lumi nuisance rather than
    aborting. Truly unknown eras still raise.
    """
    if era in ["2016", "2016APV", "2017", "2018"]:
        try:
            card.add_log_normal(p.name, f"CMS_lumi_{era}", config.luminosity.uncer)
            warnings.warn("Old style single-lumi uncertainty detected, please update to uncer_<lumi_uncertainty>"
                          "\n where <lumi_uncertainty> is one of lumi_13TeV_1516_l, lumi_13TeV_151617_l, lumi_13TeV_15161718_l"
                          "\n See https://twiki.cern.ch/twiki/bin/viewauth/CMS/LumiRecommendationsRun2 for more details")
        except Exception as e:
            pass
        try:
            card.add_log_normal(p.name, f"CMS_lumi_{era}", getattr(config.luminosity, f"uncer_lumi_{era}"))
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
    elif era in RUN3_ERAS:
        # Run 3 (13.6 TeV): add whatever the config provides; warn if nothing is there.
        # NOTE: config.luminosity.__getattr__ raises KeyError on a missing key (not
        # AttributeError), so getattr(..., None) does NOT protect us -- catch explicitly.
        def _cfg_lumi(key):
            try:
                return getattr(config.luminosity, key)
            except (KeyError, AttributeError):
                return None

        added_any = False
        unc_year = _cfg_lumi(f"uncer_lumi_{era}")
        if unc_year is not None:
            card.add_log_normal(p.name, f"CMS_lumi_13p6TeV_{era}", unc_year)
            added_any = True
        unc_corr = _cfg_lumi("uncer_lumi_13p6TeV_correlated")
        if unc_corr is not None:
            card.add_log_normal(p.name, "CMS_lumi_13p6TeV_correlated", unc_corr)
            added_any = True
        if not added_any:
            warnings.warn(
                f"[cards] Run 3 luminosity uncertainty not found in config for era "
                f"'{era}' (looked for luminosity.uncer_lumi_{era} and "
                f"luminosity.uncer_lumi_13p6TeV_correlated). Building the datacard "
                f"WITHOUT a luminosity nuisance for '{p.name}'."
            )
    else:
        raise NotImplementedError(f"Luminosity uncertainties not implemented for era '{era}'")


def plan_group(mode, boson, name, gtype, channel):
    """Per config-group plan for a given mode.
    Returns dict(card_name, channel, ptype) or None to skip this group.
    `name` stays the config-group key (for processes / remap lookup); the card
    process label is `card_name`."""
    is_split = name in POL_SPLIT_GROUPS

    if mode == "standard":
        if is_split:                             # the split templates are copies of the inclusive WZ
            return None
        return dict(card_name=name, channel=channel, ptype=gtype)

    if mode == "polarization":
        keep = {f"WZ_{boson}_{p}" for p in POLS}
        if gtype == "signal":
            if name in keep:
                return dict(card_name=name, channel=channel, ptype="signal")
            return None                          # drop inclusive WZ + other bosons
        return dict(card_name=name, channel=channel, ptype=gtype)

    if mode == "asymmetry":
        if is_split:
            return None
        if gtype == "signal":                    # inclusive WZ -> WZ_Wp / WZ_Wm
            tag = charge_tag(channel)
            return dict(card_name=f"WZ_{tag}" if tag else name,
                        channel=channel, ptype="signal")
        return dict(card_name=name, channel=channel, ptype=gtype)

    if mode == "fiducial":
        if is_split:
            return None
        if gtype == "signal":                    # WZ from the in-fiducial channel
            return dict(card_name="WZ", channel=f"{channel}_IFV", ptype="signal")
        return dict(card_name=name, channel=channel, ptype=gtype)

    if mode == "asymmetry_fiducial":
        # charge-split AND fiducial. --channel is the base charge channel
        # (e.g. inc-SR01_p); the in-fiducial signal is taken from <chan>_IFV and
        # renamed WZ_Wp / WZ_Wm so ChargeAsymmetryWZ scales it by rplus/rminus.
        # Backgrounds and data stay in the plain charge channel; the OFV piece
        # is added after the loop, exactly as in `fiducial`.
        if is_split:
            return None
        if gtype == "signal":
            tag = charge_tag(channel)
            return dict(card_name=f"WZ_{tag}" if tag else "WZ",
                        channel=f"{channel}_IFV", ptype="signal")
        return dict(card_name=name, channel=channel, ptype=gtype)

    return dict(card_name=name, channel=channel, ptype=gtype)


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
    parser.add_argument("--bins",  type=lambda s: [float(item) for item in s.split(',')], help='input a comma separated list. ex: --bins="-1.2,0,1.2"')
    parser.add_argument('--blind', action='store_true', help='blinding the channel')
    parser.add_argument('--checksyst', action='store_true')
    parser.add_argument("-rrt", "--remap_replacement_types", nargs='*', type=str, default=[])
    parser.add_argument("--mode", type=str, default="standard",
                        choices=["standard", "polarization", "asymmetry", "fiducial", "asymmetry_fiducial"],
                        help="which WZ measurement this card is for")
    parser.add_argument("--boson", type=str, default="W", choices=list(BOSON_OBS),
                        help="(polarization mode) which boson helicity to fit: W, Wp, Wm, Z, "
                             "Z_WpZ (Z in W+Z) or Z_WmZ (Z in W-Z)")

    options = parser.parse_args()
    config = dctools.read_config(options.input)

    if len(options.channel) == 1:
        options.channel = options.channel[0]

    # ---- pick the fit observable for this mode -----------------------------
    if options.mode == "polarization":
        options.variable = BOSON_OBS[options.boson]
    observable = options.variable

    print(f'making [{options.mode}]: {options.channel} : {observable} : {options.era}'
          + (f' : boson={options.boson}' if options.mode == "polarization" else ''))
    if options.mode == "asymmetry_fiducial":
        tag = charge_tag(options.channel)
        if tag is None:
            raise ValueError(
                "asymmetry_fiducial mode needs a charge-split channel, e.g. "
                f"--channel inc-SR01_p or inc-SR01_m (got '{options.channel}')")
        rich.print(f"   signal     : [green]WZ_{tag}[/green] from "
                   f"[cyan]{options.channel}_IFV[/cyan]")
        rich.print(f"   OFV bkg    : [yellow]WZ_OFV[/yellow] from "
                   f"[cyan]{options.channel}_OFV[/cyan]  (background, not scaled)")
        rich.print(f"   other bkg  : from [cyan]{options.channel}[/cyan]")

    # make datasets per process
    datasets = {}
    signal = ""

    if options.name=='':
        options.name == options.channel


    datasets:Dict = dict()
    for name in config.groups:
        gtype = config.groups[name].type
        plan = plan_group(options.mode, options.boson, name, gtype, options.channel)
        if plan is None:
            continue

        histograms = dict(
            filter(
                lambda _n: _n[0] in config.groups[name].processes,
                config.boosthist.items()
            )
        )
        rebin = get_rebin(config, plan["channel"], observable, options.rebin)

        p = dctools.datagroup(
            histograms = histograms,
            ptype      = plan["ptype"],
            observable = observable,
            name       = plan["card_name"],
            xsections  = config.xsections,
            channel    = plan["channel"],
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

    if options.mode == "polarization":
        # Unmatched WZ: the part of the inclusive WZ that is in none of the three
        # pure-state templates (no valid gen W -> tau nu + Z -> ll, e.g. fake or
        # leptonic taus). U = nominal - sum_p f_p T_p, for the nominal and every
        # systematic variation, with the generator fractions f_p recovered from the
        # same file (pol_unmatched_hist above). It is a background: the POIs only
        # scale the templates. Each piece is keyed by its parent WZ dataset name, so
        # the standard cross-section lookup applies (it is a part of that sample).
        wz_group = "WZ"
        if wz_group not in config.groups:
            raise KeyError(f"polarization mode needs the inclusive '{wz_group}' group in the "
                           f"config to build the unmatched WZ background")
        unmatched = {}
        for proc in config.groups[wz_group].processes:
            if proc not in config.boosthist:
                continue
            entry = config.boosthist[proc]
            h_u, fr = pol_unmatched_hist(entry["hist"], options.boson)
            rich.print(f"   {proc} {options.boson}: f0={fr['f0']:.4f} fL={fr['fL']:.4f} fR={fr['fR']:.4f}")
            unmatched[proc] = {"hist": {observable: h_u}, "sumw": entry["sumw"]}
        if not unmatched:
            raise KeyError(f"none of {list(config.groups[wz_group].processes)} is in the input histograms")
        um_name = f"WZ_{options.boson}_unmatched"
        p_um = dctools.datagroup(
            histograms = unmatched,
            ptype      = "background",
            observable = observable,
            name       = um_name,
            xsections  = config.xsections,
            channel    = options.channel,
            luminosity = config.luminosity.value,
            rebin      = get_rebin(config, options.channel, observable, options.rebin),
            remap_class_name = None,
        )
        datasets[um_name] = p_um
        n_tpl = ", ".join(f"{p} {datasets[f'WZ_{options.boson}_{p}'].get('nominal').sum().value:.2f}"
                          for p in POLS if f"WZ_{options.boson}_{p}" in datasets)
        rich.print(f"   [yellow]{um_name}[/yellow] (background): "
                   f"{p_um.get('nominal').sum().value:.2f} events in {options.channel}  "
                   f"(pure-state templates, each normalised to all of WZ: {n_tpl})")

    if options.mode in FIDUCIAL_MODES:
        wz_group = "WZ"
        if wz_group in config.groups:
            histograms = dict(
                filter(
                    lambda _n: _n[0] in config.groups[wz_group].processes,
                    config.boosthist.items()
                )
            )
            ofv_channel = f"{options.channel}_OFV"
            rebin = get_rebin(config, ofv_channel, observable, options.rebin)
            p_ofv = dctools.datagroup(
                histograms = histograms,
                ptype      = "background",
                observable = observable,
                name       = "WZ_OFV",
                xsections  = config.xsections,
                channel    = ofv_channel,
                luminosity = config.luminosity.value,
                rebin      = rebin,
                remap_class_name = None,
            )
            datasets["WZ_OFV"] = p_ofv


    if options.plot:
        _plot_channel = plotter.add_process_axis(datasets)
        pred = _plot_channel.project('process','systematic', observable)[:hist.loc('data'),:,:]
        data = _plot_channel[{'systematic':'nominal'}].project('process',observable)[hist.loc('data'),:] 

        plt.figure(figsize=(6,7))
        ax, bx = plotter.mcplot(
            pred[{'systematic':'nominal'}].stack('process'),
            data=None if options.blind else data,
            syst=pred.stack('process'),
        )

        try:
            sig_qcd = _plot_channel[{'systematic':'nominal'}].project('process', observable)[hist.loc('WZ'),:]
            sig_qcd.plot(ax=ax, histtype='step', color='purple')
        except:
            pass

        ymax = np.max([line.get_ydata().max() for line in ax.lines if line.get_ydata().shape[0]>0])
        ymin = np.min([line.get_ydata().min() for line in ax.lines if line.get_ydata().shape[0]>0])

        ax.set_ylim(0.001, 100*ymax)
        ax.set_title(f"channel {options.channel}: {options.era}")

        ax.set_yscale('log')
        plt.savefig(f'plot-{options.channel}-{observable}-{options.era}.pdf')


    if options.checksyst:
        _plot_channel = plotter.add_process_axis(datasets)
        pred = _plot_channel.project('process','systematic', observable)[:hist.loc('data'),:,:]
        data = _plot_channel[{'systematic':'nominal'}].project('process',observable)[hist.loc('data'),:]
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

    # ------------------------------------------------------------------
    # Nuisances are driven by the histograms: every variation on a process's
    # 'systematic' axis becomes a shape nuisance. NUISANCE_NAMES only renames
    # the processor's names to the CMS datacard names; anything not listed keeps
    # its own name (the JES sources, prefiring_weight, jetPUid_sf, HEM, the DD
    # stat_<era>, ...). MC and the data-driven estimate use the SAME table, so a
    # variation propagated into the DD estimate always gets the same name as the
    # MC one and the two stay correlated.
    # Systematics conventions: https://gitlab.cern.ch/cms-analysis/general/systematics/-/blob/master/systematics_master.yml
    # ------------------------------------------------------------------
    era = options.era
    NUISANCE_NAMES = {
        "scale_e":       f"CMS_scale_e_{era}",
        "res_e":         f"CMS_res_e_{era}",
        "scale_m":       f"CMS_scale_m_{era}",
        "res_m":         f"CMS_res_m_{era}",
        "scale_t":       f"CMS_scale_t_{era}",
        "eff_e_reco":    f"CMS_eff_e_reco_{era}",
        "eff_e_id":      f"CMS_eff_e_id_{era}",
        "eff_m_id":      f"CMS_eff_m_id_{era}",
        "eff_m_iso":     f"CMS_eff_m_iso_{era}",
        "tauIDvsjet_sf": f"CMS_t_id_vsjet_{era}",
        "tauIDvse_sf":   f"CMS_t_id_vse_{era}",
        "tauIDvsmu_sf":  f"CMS_t_id_vsmu_{era}",
        "triggerSF":     f"CMS_trig_sf_{era}",
        "pileup_weight": f"CMS_pileup_{era}",
        "JER":           f"CMS_jer_{era}",
        "UES":           f"CMS_UES_{era}",
        "UEPS_FSR":      "PS_FSR",
        "UEPS_ISR":      "PS_ISR",
        "PDF_weight":    "pdf",      # might have to be anticorrelated for W+Z and W-Z
        "aS_weight":     "alphaS",
    }
    # Variations handled outside the plain rename:
    #   QCDScale0w/1w/2w -> collapsed into one QCD-scale nuisance per process
    #   JES              -> dropped: the total JES, already covered by the split JES sources
    #   kEW              -> ewk_corr_WZ / ewk_corr_ZZ for the WZ / ZZ MC, dropped otherwise
    #                       (the EWK correction does not apply to the data-driven DY)
    QCD_BASES = ("QCDScale0w", "QCDScale1w", "QCDScale2w")
    DROPPED_BASES = {"JES"}

    def qcd_scale_name(p):
        if p.remap_replace_group_name is not None:
            return "CMS_QCDScale_DDDY_prop"
        # the polarization templates and the unmatched part are all pieces of one
        # WZ sample: one QCD-scale nuisance for all of them, not one per piece
        return "CMS_QCDScaleWZ" if p.name in WZ_POL_PIECES else f"CMS_QCDScale{p.name}"

    def ewk_name(p):
        if p.remap_replace_group_name is not None:
            return None
        if "WZ" in p.name:
            return "ewk_corr_WZ"
        if "ZZ" in p.name and "EWK" not in p.name:
            return "ewk_corr_ZZ"
        return None

    for _, p in datasets.items():
        print(" --> ", p.name)
        if len(p.to_boost().shape) == 0 or p.get("nominal").sum().value == 0:
            print(f"--> histogram for the process {p.name} is empty !")
            continue

        if p.ptype=="data":
            continue
        if not card.add_nominal(p.name, p.get("nominal"), p.ptype): continue
        is_dd = p.remap_replace_group_name is not None

        # luminosity (Run 2 scheme, Run 3 warn-and-continue, else raise); not for the
        # data-driven estimate, which comes from data and does not scale with the luminosity
        if not is_dd:
            add_lumi_uncertainties(card, p, config, options.era)

        present = set(_shape_nuisance_bases(p))
        added, dropped = [], []
        for base in sorted(present):
            if base in QCD_BASES:
                continue                                   # handled together below
            if base in DROPPED_BASES:
                dropped.append(base)
                continue
            if base == "kEW":
                nuis = ewk_name(p)
                if nuis is None:
                    dropped.append(base)
                    continue
            else:
                nuis = NUISANCE_NAMES.get(base, base)
            card.add_shape_nuisance(p.name, nuis, p.get(base), symmetrise=False)
            added.append(f"{base}->{nuis}" if nuis != base else base)

        if all(b in present for b in QCD_BASES):
            nuis = qcd_scale_name(p)
            card.add_qcd_scales(p.name, nuis, [p.get(b) for b in QCD_BASES])
            added.append(f"QCDScale0w/1w/2w->{nuis}")
        elif present & set(QCD_BASES):
            warnings.warn(f"[cards] {p.name}: partial QCDScale set {sorted(present & set(QCD_BASES))}; "
                          f"skipping QCD scale nuisance.")

        if is_dd:
            rich.print(f"[green]DD {p.name} (from {p.remap_original_group_name}): added {added}")
            if dropped:
                rich.print(f"[yellow]DD {p.name}: dropped {dropped}")
        else:
            # Add Barlow-Beeston Lite MC Stat uncertainty
            if ('DY' not in p.name):
                card.add_auto_stat()

    # saving the datacard
    card.dump()


if __name__ == "__main__":
    main()