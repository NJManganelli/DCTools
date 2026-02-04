import warnings
import traceback

class inc_WZ_DataDrivenDY:
    @staticmethod
    def replaces_group_and_type(dc_instance):
        mapping = {
            "inc-D0": ("DY", "validation"),
            "inc-D1": ("DY", "validation"),
            "inc-SR0": ("DY", "datadriven"),
            "inc-SR1": ("DY", "datadriven"),
            "inc-SR01": ("DY", "datadriven"),
        }
        assert all([chan in mapping.keys() for chan in inc_WZ_DataDrivenDY.expected_channels(dc_instance)]), f"In {__class__.__name__}, not all expected channels are found in the replaces_group_and_type method: {mapping.keys()}"
        return mapping

    @staticmethod
    def expected_channels(dc_instance):
        return ['inc-D0', 'inc-SR0', 'inc-D1', 'inc-SR1', 'inc-SR01',]

    @staticmethod
    def expected_systvars(dc_instance):
        return ["nominal", "DDDYUp", "DDDYDown"]
    
    @staticmethod
    def expected_channels_systvars(dc_instance):
        # simple mapping
        return [(chan, syst) for chan in inc_WZ_DataDrivenDY.expected_channels(dc_instance) for syst in inc_WZ_DataDrivenDY.expected_systvars(dc_instance)]

    @staticmethod
    def skip_scale(dc_instance):
        return True

    @staticmethod
    def channel_syst_mapping(expected_channel=None, expected_systvar=None):
        # The mapping function for the DataDriven DY, mapping Region "B0"/"B1" to "SR0"/"SR1" respectively, and "C0"/"C1" to "D0"/"D1" respectively (for closure testing)
        # Simultaneously maps the expected systvar to the name generated while filling histograms
        # Simple mapping, channel-to-channel and systvar-to-systvar, but by mapping together we can accommodate a region-specific systematic remapping too
        channel_dict = {
            # maps the expected channel (i.e. for analysis) to the original filled channel (i.e. SR0:nominal pulls from B0:datadriven-DDDYNominalUp in data for the data-driven DY estimate)
            "inc-SR0": "inc-B0",
            "inc-SR1": "inc-B1",
            "inc-SR01": "inc-B01",
            "inc-D0": "inc-C0",
            "inc-D1": "inc-C1",
        }
        syst_dict = {
            "nominal": "datadriven_DDDYNominalUp",
            "DDDYUp": "datadriven_DDDYUp",
            "DDDYDown": "datadriven_DDDYDown",
        }
        assert (expected_channel, expected_systvar) in inc_WZ_DataDrivenDY.expected_channels_systvars(None), f"In {__class__.__name__}, the expected (channel, systvar) pair ({expected_channel}, {expected_systvar}) is not found in the permitted expected_channels_systvars ({inc_WZ_DataDrivenDY.expected_channels_systvars(None)})"
        #assert expected_systvar in expected_systvars(None), f"In {__class__.__name__}, expected_systvar {expected_systvar} not found in permitted expected_systvars ({expected_systvars(None)})"
        #assert expected_channel in expected_channels(None), f"In {__class__.__name__}, expected_channel {expected_channel} not found in permitted expected_channel ({expected_channels(None)})"

        return {"channel": channel_dict[expected_channel], "systematic": syst_dict[expected_systvar]}

    
    @staticmethod
    def histograms(dc_instance):
        import hist
        import copy
        modified_histograms = copy.deepcopy(dc_instance.histograms)
        for proc, hist_and_sumw in dc_instance.histograms.items():
            hist_data = hist_and_sumw['hist']
            orig_axes = hist_data.axes
            assert "systematic" in orig_axes.name, f"In {__class__.__name__}, the histograms must have a 'systematic' axis to use the remap_class. Found axes: {orig_axes.name}"
            assert "channel" in orig_axes.name, f"In {__class__.__name__}, the histograms must have a 'channel' axis to use the remap_class. Found axes: {orig_axes.name}"
            assert dc_instance.observable in orig_axes.name, f"In {__class__.__name__}, the histograms are expected to have the observable axis '{dc_instance.observable}' to use the remap_class. Found axes: {orig_axes.name}"
            new_axes = []
            for ax in orig_axes:
                if ax.name == "systematic":
                    new_ax = hist.axis.StrCategory(inc_WZ_DataDrivenDY.expected_systvars(dc_instance), name="systematic", growth=ax.traits.growth, overflow=ax.traits.overflow)
                elif ax.name == "channel":
                    new_ax = hist.axis.StrCategory(inc_WZ_DataDrivenDY.expected_channels(dc_instance), name="channel", growth=ax.traits.growth, overflow=ax.traits.overflow)
                else:
                    new_ax = ax
                new_axes.append(new_ax)
            new_hist = hist.Hist(*new_axes, storage=hist_data.storage_type())
            # Now we fill the new histogram by mapping the expected channel/syst to the original filled channel/syst
            for expected_channel, expected_systvar in inc_WZ_DataDrivenDY.expected_channels_systvars(dc_instance):
                # select the original histogram slice
                try:
                    mapping = inc_WZ_DataDrivenDY.channel_syst_mapping(expected_channel=expected_channel, expected_systvar=expected_systvar)
                    orig_slice = hist_data[{"channel": mapping["channel"], "systematic": mapping["systematic"]}]
                except Exception as e:
                    try:
                        mapping = inc_WZ_DataDrivenDY.channel_syst_mapping(expected_channel=expected_channel, expected_systvar=expected_systvar)
                        if mapping["systematic"] not in hist_data.axes['systematic']:
                            warnings.warn(traceback.format_exc())
                    except Exception as e2:
                        pass
                    #warnings.warn(f"[WARNING] In {__class__.__name__}, unable to find the original histogram slice for expected (channel, systvar) = ({expected_channel}, {expected_systvar}) mapped to (channel, systvar) = ({mapping['channel']}, {mapping['systematic']}). The available channels are {list(hist_data.axes['channel'])} and the available systematics are {list(hist_data.axes['systematic'])}. Filling with zeroed histogram. datagroup={print(dc_instance)} Exception: {e}")
                    orig_slice = hist_data[{"channel": 0, "systematic": 0}].copy().reset()
                # fill the new histogram slice
                new_hist[{"channel": expected_channel, "systematic": expected_systvar}] = orig_slice.view(flow=True)
            modified_histograms[proc]['hist'] = new_hist
        return modified_histograms

remap_master_class_dict = {}
remap_master_class_dict["inc_WZ_DataDrivenDY"] = inc_WZ_DataDrivenDY

for key, func in remap_master_class_dict.items():
    assert hasattr(func, "replaces_group_and_type") and callable(func.replaces_group_and_type) and len(func.replaces_group_and_type(None).keys()) > 0, f"Remap function {key} is missing the `replaces_group_and_type` method, which is required and returns the channel mapping of replacement process-group-name+replacement-type, e.g. datadriven or validation"
    assert hasattr(func, "skip_scale") and callable(func.skip_scale) and (func.skip_scale(None) in [False, True]), f"Remap function {key} is missing the 'skip_scale' method, which is required and must return a boolean value."
    assert hasattr(func, "histograms") and callable(func.histograms), f"Remap function {key} is missing the 'histograms' method, which is required and must be callable."