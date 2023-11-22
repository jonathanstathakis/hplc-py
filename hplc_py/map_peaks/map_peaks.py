import numpy as np
import warnings

class PeakMapper:
    
    def map_peaks(self,
                  params,
                  loc_tolerance=0.5,
                  include_unmapped=False):
        R"""
        Maps user-provided mappings to arbitrarily labeled peaks. If a linear 
        calibration curve is also provided, the concentration will be computed.

        Parameters
        ----------
        params : `dict` 
            A dictionary mapping each peak to a slope and intercept used for 
            converting peak areas to units of concentraions. Each peak should
            have a key that is the compound name (e.g. "glucose"). Each key
            should have another dict as the key with `retention_time` , `slope` ,
            and `intercept` as keys. If only `retention_time` is given,
            concentration will not be computed. The key `retention_time` will be
            used to map the compound to the `peak_id`. If `unit` are provided,
            this will be added as a column
       loc_tolerance : `float`
           The tolerance for mapping the compounds to the retention time. The 
           default is 0.5 time units.
       include_unmapped : `bool`
            If True, unmapped compounds will remain in the returned peak dataframe,
            but will be populated with Nan. Default is False.

       Returns
       -------
       peaks : `pandas.core.frame.DataFrame`
            A modified peak table with the compound name and concentration 
            added as columns.

        Notes
        -----
        .. note::
            As of `v0.1.0`, this function can only accommodate linear calibration 
            functions.
        """
        # Create a mapper for peak id to compound
        mapper = {}
        unmapped = {}
        peak_df = self.peaks.copy()
        for k, v in params.items():
            ret_time = v['retention_time']
            peak_id = np.abs(
                peak_df['retention_time'].values - ret_time) < loc_tolerance

            if np.sum(peak_id) > 1:
                raise ValueError(
                    f"Multiple compounds found within tolerance of retention time for {k}. Reduce the tolerance or correct the provided value.")

            if np.sum(peak_id) == 0:
                unmapped[k] = v['retention_time']
                break
            peak_id = peak_df.peak_id.values[np.argmax(peak_id)]
            peak_df.loc[peak_df['peak_id'] == peak_id, 'compound'] = k
            mapper[peak_id] = k
        if len(mapper) == 0:
            raise ValueError(
                "No peaks could be properly mapped! Check your provided retention times.")
        if len(unmapped) > 0:
            for k, v in unmapped.items():
                warnings.warn(
                    f"\nNo peak found for {k} (retention time {v}) within the provided tolerance.")

        # Iterate through the compounds and calculate the concentration.
        for g, d in peak_df.groupby('compound'):
            if (g in params.keys()):
                if 'slope' in params[g].keys():
                    conc = (d['area'] - params[g]['intercept']) / \
                        params[g]['slope']
                    peak_df.loc[peak_df['compound']
                                == g, 'concentration'] = conc
                    if 'unit' in params[g].keys():
                        peak_df.loc[peak_df['compound'] ==
                                    g, 'unit'] = params[g]['unit']
        if include_unmapped == False:
            peak_df.dropna(inplace=True)
        self.quantified_peaks = peak_df
        self._mapped_peaks = mapper
        return peak_df