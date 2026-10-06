import warnings


class inc_WZ_DataDrivenDY:
    # Set to True to get a one-line-per-process report of channels/systematics
    # that had no source. The old code warned with a full traceback for EVERY
    # (channel, systematic) miss, which for 37 expected channels buried
    # everything else in the log.
    verbose = False

    @staticmethod
    def replaces_group_and_type(dc_instance):
        mapping = {
            "inc-SR01": ("DY", "datadriven"),
            "inc-SR01_IFV": ("DY", "datadriven"),
            "inc-SR01_OFV": ("DY", "datadriven"),
            "inc-SR01_p": ("DY", "datadriven"),
            "inc-SR01_p_IFV": ("DY", "datadriven"),
            "inc-SR01_p_OFV": ("DY", "datadriven"),
            "inc-SR01_m": ("DY", "datadriven"),
            "inc-SR01_m_IFV": ("DY", "datadriven"),
            "inc-SR01_m_OFV": ("DY", "datadriven"),
            "inc-SR0": ("DY", "datadriven"),
            "inc-SR0_p": ("DY", "datadriven"),
            "inc-SR0_m": ("DY", "datadriven"),
            "inc-SR0+": ("DY", "datadriven"),
            "inc-SR0-": ("DY", "datadriven"),
            "inc-SR1": ("DY", "datadriven"),
            "inc-SR1_p": ("DY", "datadriven"),
            "inc-SR1_m": ("DY", "datadriven"),
            "inc-SR1+": ("DY", "datadriven"),
            "inc-SR1-": ("DY", "datadriven"),
            "inc-VR01": ("DY", "datadriven"),
            "inc-VR01_p": ("DY", "datadriven"),
            "inc-VR01_m": ("DY", "datadriven"),
            "inc-VR0": ("DY", "datadriven"),
            "inc-VR0+": ("DY", "datadriven"),
            "inc-VR0-": ("DY", "datadriven"),
            "inc-VR1": ("DY", "datadriven"),
            "inc-VR1+": ("DY", "datadriven"),
            "inc-VR1-": ("DY", "datadriven"),
            "inc-D01": ("DY", "validation"),
            "inc-D01_p": ("DY", "validation"),
            "inc-D01_m": ("DY", "validation"),
            "inc-D0": ("DY", "validation"),
            "inc-D0+": ("DY", "validation"),
            "inc-D0-": ("DY", "validation"),
            "inc-D1": ("DY", "validation"),
            "inc-D1+": ("DY", "validation"),
            "inc-D1-": ("DY", "validation"),
        }
        assert all([chan in mapping.keys() for chan in inc_WZ_DataDrivenDY.expected_channels(dc_instance)]), f"In {__class__.__name__}, not all expected channels are found in the replaces_group_and_type method: {mapping.keys()}"
        return mapping

    @staticmethod
    def expected_channels(dc_instance):
        return ['inc-SR01', 'inc-SR01_IFV', 'inc-SR01_OFV', 'inc-SR01_p', 'inc-SR01_p_IFV',
                'inc-SR01_p_OFV', 'inc-SR01_m', 'inc-SR01_m_IFV', 'inc-SR01_m_OFV', 'inc-SR0',
                'inc-SR0_p', 'inc-SR0_m', 'inc-SR0+', 'inc-SR0-', 'inc-SR1', 'inc-SR1_p',
                'inc-SR1_m', 'inc-SR1+', 'inc-SR1-', 'inc-VR01', 'inc-VR01_p', 'inc-VR01_m',
                'inc-VR0', 'inc-VR0+', 'inc-VR0-', 'inc-VR1', 'inc-VR1+', 'inc-VR1-', 'inc-D01',
                'inc-D01_p', 'inc-D01_m', 'inc-D0', 'inc-D0+', 'inc-D0-', 'inc-D1', 'inc-D1+',
                'inc-D1-']

    # ------------------------------------------------------------------
    # Systematics are DISCOVERED from the filled histograms rather than
    # hardcoded. The processor emits a conditional set:
    #   * always   datadriven_DDDYNominalUp/Down  (nominal added twice)
    #   * EITHER   datadriven_stat_<era>Up/Down   (stat_systematics branch)
    #     OR       datadriven_DDDYUp/Down         (legacy branch)
    #   * plus a variable number of MC-propagated nuisances carried under their
    #     BARE source names, so they correlate with the same-named MC nuisance.
    # Hardcoding ["nominal","DDDYUp","DDDYDown"] silently drops the per-era stat
    # and every propagated MC nuisance, and on the stat branch it drops the DD
    # uncertainty entirely -- leaving the dominant background with no systematic.
    # ------------------------------------------------------------------

    # the DD central estimate (the nominal weight was added twice in the processor)
    NOMINAL_SOURCE = "datadriven_DDDYNominalUp"

    @staticmethod
    def _source_systematics(dc_instance):
        """Union of the 'systematic' categories present across the source histograms."""
        names = set()
        for hist_and_sumw in dc_instance.histograms.values():
            names.update(map(str, hist_and_sumw['hist'].axes['systematic']))
        return names

    @staticmethod
    def _syst_output_to_source(dc_instance):
        """{output_systvar (datacard) : source_systvar (as filled)}.

        The DD central estimate lives in datadriven_DDDYNominalUp (the nominal
        weight was added twice), so it becomes the output 'nominal' and the
        redundant ...NominalDown is dropped. DD-internal nuisances keep their
        name minus the 'datadriven_' prefix; MC-propagated ones are already bare
        and pass through unchanged.
        """
        src = inc_WZ_DataDrivenDY._source_systematics(dc_instance)
        assert inc_WZ_DataDrivenDY.NOMINAL_SOURCE in src, (
            f"In {__class__.__name__}, the source histograms are missing "
            f"'{inc_WZ_DataDrivenDY.NOMINAL_SOURCE}' (the DD central estimate). Found: {sorted(src)}"
        )
        mapping = {"nominal": inc_WZ_DataDrivenDY.NOMINAL_SOURCE}
        for s in sorted(src):
            if s in ("nominal", "datadriven_DDDYNominalUp", "datadriven_DDDYNominalDown"):
                continue
            out = s[len("datadriven_"):] if s.startswith("datadriven_") else s
            mapping[out] = s
        return mapping

    @staticmethod
    def expected_systvars(dc_instance):
        # Called with None only for validation asserts; return the invariant minimum.
        if dc_instance is None:
            return ["nominal"]
        return list(inc_WZ_DataDrivenDY._syst_output_to_source(dc_instance).keys())

    @staticmethod
    def expected_channels_systvars(dc_instance):
        return [(chan, syst)
                for chan in inc_WZ_DataDrivenDY.expected_channels(dc_instance)
                for syst in inc_WZ_DataDrivenDY.expected_systvars(dc_instance)]

    @staticmethod
    def skip_scale(dc_instance):
        return True

    @staticmethod
    def channel_syst_mapping(dc_instance=None, expected_channel=None, expected_systvar=None):
        # Region "B0"/"B1" -> "SR0"/"SR1", "VB*" -> "VR*", "C0"/"C1" -> "D0"/"D1" (closure).
        #
        # The charge suffix (_p/_m/+/-) propagates to the B region; the fiducial
        # suffix (_IFV/_OFV) does NOT, because the data-driven estimate is built
        # from DATA, which has no generator-level in/out-of-fiducial split. Both
        # FV variants therefore read the same source region.
        channel_dict = {
            # --- inc-SR01 family ---
            "inc-SR01": "inc-B01",
            "inc-SR01_IFV": "inc-B01",
            "inc-SR01_OFV": "inc-B01",
            "inc-SR01_p": "inc-B01_p",
            "inc-SR01_p_IFV": "inc-B01_p",
            "inc-SR01_p_OFV": "inc-B01_p",
            "inc-SR01_m": "inc-B01_m",
            "inc-SR01_m_IFV": "inc-B01_m",
            "inc-SR01_m_OFV": "inc-B01_m",
            # --- inc-SR0 family ---
            "inc-SR0": "inc-B0",
            "inc-SR0_p": "inc-B0_p",
            "inc-SR0_m": "inc-B0_m",
            "inc-SR0+": "inc-B0+",
            "inc-SR0-": "inc-B0-",
            # --- inc-SR1 family ---
            "inc-SR1": "inc-B1",
            "inc-SR1_p": "inc-B1_p",
            "inc-SR1_m": "inc-B1_m",
            "inc-SR1+": "inc-B1+",
            "inc-SR1-": "inc-B1-",
            # --- inc-VR01 family ---
            "inc-VR01": "inc-VB01",
            "inc-VR01_p": "inc-VB01_p",
            "inc-VR01_m": "inc-VB01_m",
            # --- inc-VR0 family ---
            "inc-VR0": "inc-VB0",
            "inc-VR0+": "inc-VB0+",
            "inc-VR0-": "inc-VB0-",
            # --- inc-VR1 family ---
            "inc-VR1": "inc-VB1",
            "inc-VR1+": "inc-VB1+",
            "inc-VR1-": "inc-VB1-",
            # --- inc-D01 family ---
            "inc-D01": "inc-C01",
            "inc-D01_p": "inc-C01_p",
            "inc-D01_m": "inc-C01_m",
            # --- inc-D0 family ---
            "inc-D0": "inc-C0",
            "inc-D0+": "inc-C0+",
            "inc-D0-": "inc-C0-",
            # --- inc-D1 family ---
            "inc-D1": "inc-C1",
            "inc-D1+": "inc-C1+",
            "inc-D1-": "inc-C1-",
        }
        if dc_instance is None:
            # validation-only path: the static legacy names
            syst_dict = {
                "nominal": "datadriven_DDDYNominalUp",
                "DDDYUp": "datadriven_DDDYUp",
                "DDDYDown": "datadriven_DDDYDown",
            }
        else:
            syst_dict = inc_WZ_DataDrivenDY._syst_output_to_source(dc_instance)
        assert (expected_channel, expected_systvar) in inc_WZ_DataDrivenDY.expected_channels_systvars(dc_instance), (
            f"In {__class__.__name__}, the expected (channel, systvar) pair "
            f"({expected_channel}, {expected_systvar}) is not found in the permitted "
            f"expected_channels_systvars"
        )
        assert expected_channel in channel_dict, (
            f"In {__class__.__name__}, channel '{expected_channel}' is in expected_channels "
            f"but has no source region in channel_dict ({sorted(channel_dict)})"
        )
        return {"channel": channel_dict[expected_channel], "systematic": syst_dict[expected_systvar]}

    @staticmethod
    def histograms(dc_instance):
        import hist
        import copy
        modified_histograms = copy.deepcopy(dc_instance.histograms)
        expected_systvars = inc_WZ_DataDrivenDY.expected_systvars(dc_instance)
        expected_channels = inc_WZ_DataDrivenDY.expected_channels(dc_instance)
        pairs = inc_WZ_DataDrivenDY.expected_channels_systvars(dc_instance)

        for proc, hist_and_sumw in dc_instance.histograms.items():
            hist_data = hist_and_sumw['hist']
            orig_axes = hist_data.axes
            assert "systematic" in orig_axes.name, f"In {__class__.__name__}, the histograms must have a 'systematic' axis to use the remap_class. Found axes: {orig_axes.name}"
            assert "channel" in orig_axes.name, f"In {__class__.__name__}, the histograms must have a 'channel' axis to use the remap_class. Found axes: {orig_axes.name}"
            assert dc_instance.observable in orig_axes.name, f"In {__class__.__name__}, the histograms are expected to have the observable axis '{dc_instance.observable}' to use the remap_class. Found axes: {orig_axes.name}"

            # Resolve once, up front, instead of discovering misses via exceptions
            # inside the fill loop -- that is what produced a traceback warning per
            # (channel, systematic) pair.
            src_channels = set(map(str, hist_data.axes['channel']))
            src_systs = set(map(str, hist_data.axes['systematic']))

            new_axes = []
            for ax in orig_axes:
                if ax.name == "systematic":
                    new_ax = hist.axis.StrCategory(expected_systvars, name="systematic", growth=ax.traits.growth, overflow=ax.traits.overflow)
                elif ax.name == "channel":
                    new_ax = hist.axis.StrCategory(expected_channels, name="channel", growth=ax.traits.growth, overflow=ax.traits.overflow)
                else:
                    new_ax = ax
                new_axes.append(new_ax)
            new_hist = hist.Hist(*new_axes, storage=hist_data.storage_type())

            missing_channels, missing_systs = set(), set()
            for expected_channel, expected_systvar in pairs:
                mapping = inc_WZ_DataDrivenDY.channel_syst_mapping(
                    dc_instance, expected_channel=expected_channel, expected_systvar=expected_systvar)
                src_c, src_s = mapping["channel"], mapping["systematic"]

                if src_c not in src_channels:
                    # No events of this data stream in the source region: its
                    # contribution is zero for every systematic. The zero slice is
                    # built from new_hist, whose axes always contain this channel/
                    # systvar -- indexing hist_data at position 0 fails with
                    # IndexError when a process has an empty category axis, which
                    # is what crashed the 2018 pol cards.
                    missing_channels.add(src_c)
                    orig_slice = new_hist[{"channel": expected_channel, "systematic": expected_systvar}].copy().reset()
                else:
                    if src_s not in src_systs:
                        # This data stream does not carry the variation (another
                        # stream does), so it is unaffected by it: use its nominal.
                        # A zero here would remove the stream's whole contribution
                        # from the variation and read as a -100% shift of it.
                        missing_systs.add(src_s)
                        src_s = inc_WZ_DataDrivenDY.NOMINAL_SOURCE
                    orig_slice = hist_data[{"channel": src_c, "systematic": src_s}]

                new_hist[{"channel": expected_channel, "systematic": expected_systvar}] = orig_slice.view(flow=True)

            if inc_WZ_DataDrivenDY.verbose and (missing_channels or missing_systs):
                # one concise line per process, not one traceback per pair
                warnings.warn(
                    f"{__class__.__name__} [{proc}]: "
                    f"{len(missing_channels)} source channel(s) absent (zero) {sorted(missing_channels)}; "
                    f"{len(missing_systs)} systematic(s) absent (nominal used) {sorted(missing_systs)}"
                )
            modified_histograms[proc]['hist'] = new_hist
        return modified_histograms


remap_master_class_dict = {}
remap_master_class_dict["inc_WZ_DataDrivenDY"] = inc_WZ_DataDrivenDY

for key, func in remap_master_class_dict.items():
    assert hasattr(func, "replaces_group_and_type") and callable(func.replaces_group_and_type) and len(func.replaces_group_and_type(None).keys()) > 0, f"Remap function {key} is missing the `replaces_group_and_type` method, which is required and returns the channel mapping of replacement process-group-name+replacement-type, e.g. datadriven or validation"
    assert hasattr(func, "skip_scale") and callable(func.skip_scale) and (func.skip_scale(None) in [False, True]), f"Remap function {key} is missing the 'skip_scale' method, which is required and must return a boolean value."
    assert hasattr(func, "histograms") and callable(func.histograms), f"Remap function {key} is missing the 'histograms' method, which is required and must be callable."
