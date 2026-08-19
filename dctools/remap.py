import warnings
import traceback

class inc_WZ_DataDrivenDY:
    @staticmethod
    def replaces_group_and_type(dc_instance):
        mapping = {
            "inc-D01": ("DY", "validation"),
            "inc-D0": ("DY", "validation"),
            "inc-D1": ("DY", "validation"),
            "inc-SR0": ("DY", "datadriven"),
            "inc-SR1": ("DY", "datadriven"),
            "inc-VR01": ("DY", "datadriven"),
            "inc-VR1": ("DY", "datadriven"),
            "inc-SR01": ("DY", "datadriven"),
        }
        assert all([chan in mapping.keys() for chan in inc_WZ_DataDrivenDY.expected_channels(dc_instance)]), f"In {__class__.__name__}, not all expected channels are found in the replaces_group_and_type method: {mapping.keys()}"
        return mapping

    @staticmethod
    def expected_channels(dc_instance):
        return ['inc-D01', 'inc-SR0', 'inc-D0', 'inc-D1', 'inc-SR1', 'inc-VR01', 'inc-VR1', 'inc-SR01',]

    # ------------------------------------------------------------------
    # Systematics are now DISCOVERED from the filled histograms instead of
    # being hardcoded, because the processor emits a conditional / list-driven
    # set: always datadriven_DDDYNominalUp/Down; then EITHER per-era
    # datadriven_stat_<era>Up/Down (stat_systematics branch) OR the legacy
    # datadriven_DDDYUp/Down; plus a variable number of MC-propagated nuisances
    # carried under their BARE source names (no datadriven_ prefix).
    # ------------------------------------------------------------------

    @staticmethod
    def _source_systematics(dc_instance):
        """Union of the 'systematic' categories present across the source histograms."""
        names = set()
        for hist_and_sumw in dc_instance.histograms.values():
            names.update(map(str, hist_and_sumw['hist'].axes['systematic']))
        return names

    @staticmethod
    def _syst_output_to_source(dc_instance):
        """
        Build {output_systvar (datacard) : source_systvar (as filled)}.

        Convention:
          * The DD central estimate lives in datadriven_DDDYNominalUp (the nominal
            weight was added twice), so it becomes the output 'nominal'; the
            redundant datadriven_DDDYNominalDown is dropped.
          * DD-internal nuisances keep their name minus the 'datadriven_' prefix:
              - per-era stat:     datadriven_stat_<era>Up/Down -> stat_<era>Up/Down
              - legacy combined:  datadriven_DDDYUp/Down       -> DDDYUp/Down
          * MC-propagated nuisances are already bare and pass through unchanged so
            they correlate with the same-named analysis nuisance on the MC.
        """
        src = inc_WZ_DataDrivenDY._source_systematics(dc_instance)
        assert "datadriven_DDDYNominalUp" in src, (
            f"In {__class__.__name__}, the source histograms are missing "
            f"'datadriven_DDDYNominalUp' (the DD central estimate). Found: {sorted(src)}"
        )
        mapping = {"nominal": "datadriven_DDDYNominalUp"}
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
    def channel_syst_mapping(dc_instance, expected_channel=None, expected_systvar=None):
        # Maps Region "B0"/"B1" -> "SR0"/"SR1" and "C0"/"C1" -> "D0"/"D1" (closure),
        # and the expected systvar -> the name generated while filling histograms.
        # NOTE: signature now takes dc_instance first, since the syst map is dynamic.
        channel_dict = {
            "inc-SR0": "inc-B0",
            "inc-SR1": "inc-B1",
            "inc-SR01": "inc-B01",
            "inc-D01": "inc-C01",
            "inc-D0": "inc-C0",
            "inc-D1": "inc-C1",
            "inc-VR01": "inc-VB01",
            "inc-VR1": "inc-VB1",
        }
        syst_dict = inc_WZ_DataDrivenDY._syst_output_to_source(dc_instance)
        assert (expected_channel, expected_systvar) in inc_WZ_DataDrivenDY.expected_channels_systvars(dc_instance), (
            f"In {__class__.__name__}, the expected (channel, systvar) pair "
            f"({expected_channel}, {expected_systvar}) is not found in the permitted "
            f"expected_channels_systvars ({inc_WZ_DataDrivenDY.expected_channels_systvars(dc_instance)})"
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
            # Fill the new histogram by mapping each expected channel/syst to the original filled channel/syst
            for expected_channel, expected_systvar in pairs:
                try:
                    mapping = inc_WZ_DataDrivenDY.channel_syst_mapping(dc_instance, expected_channel=expected_channel, expected_systvar=expected_systvar)
                    orig_slice = hist_data[{"channel": mapping["channel"], "systematic": mapping["systematic"]}]
                except Exception as e:
                    try:
                        mapping = inc_WZ_DataDrivenDY.channel_syst_mapping(dc_instance, expected_channel=expected_channel, expected_systvar=expected_systvar)
                        if mapping["systematic"] not in list(map(str, hist_data.axes['systematic'])):
                            warnings.warn(traceback.format_exc())
                    except Exception as e2:
                        pass
                    # Fallback: zero-filled slice (kept from original behavior)
                    orig_slice = hist_data[{"channel": 0, "systematic": 0}].copy().reset()
                new_hist[{"channel": expected_channel, "systematic": expected_systvar}] = orig_slice.view(flow=True)
            modified_histograms[proc]['hist'] = new_hist
        return modified_histograms

remap_master_class_dict = {}
remap_master_class_dict["inc_WZ_DataDrivenDY"] = inc_WZ_DataDrivenDY

for key, func in remap_master_class_dict.items():
    assert hasattr(func, "replaces_group_and_type") and callable(func.replaces_group_and_type) and len(func.replaces_group_and_type(None).keys()) > 0, f"Remap function {key} is missing the `replaces_group_and_type` method, which is required and returns the channel mapping of replacement process-group-name+replacement-type, e.g. datadriven or validation"
    assert hasattr(func, "skip_scale") and callable(func.skip_scale) and (func.skip_scale(None) in [False, True]), f"Remap function {key} is missing the 'skip_scale' method, which is required and must return a boolean value."
    assert hasattr(func, "histograms") and callable(func.histograms), f"Remap function {key} is missing the 'histograms' method, which is required and must be callable."