import scipy
import numpy as np
import numpy.typing as npt

class SkewNorms:
    def _compute_skewnorm(self,
                          x,
                          *params)->npt.NDArray[np.float64]:
        R"""
        Computes the lineshape of a skew-normal distribution given the shape,
        location, and scale parameters

        Parameters
        ----------
        x : `float` or `numpy.ndarray`
            The time dimension of the skewnorm 
        params : `list`, [`amplitude`, `loc`, `scale`, `alpha`]
            Parameters for the shape and scale parameters of the skewnorm 
            distribution.
                `amplitude` : positive `float`
                    Height of the peak.
                `loc` : positive `float`
                    The location parameter of the distribution.
                `scale` : positive `float`
                    The scale parameter of the distribution.
                `alpha` : positive `float`
                    The skew shape parameter of the distribution.

        Returns
        -------
        scaled_pdf : `float or numpy array, same shape as `x`
            The PDF of the skew-normal distribution scaled with the supplied 
            amplitude.

        Notes
        -----
        This function infers the parameters defining skew-normal distributions 
        for each peak in the chromatogram. The fitted distribution has the form 

        .. math:: 
            I = 2S_\text{max} \left(\frac{1}{\sqrt{2\pi\sigma^2}}\right)e^{-\frac{(t - r_t)^2}{2\sigma^2}}\left[1 + \text{erf}\frac{\alpha(t - r_t)}{\sqrt{2\sigma^2}}\right]

        where :math:`S_\text{max}` is the maximum signal of the peak, 
        :math:`t` is the time, :math:`r_t` is the retention time, :math:`\sigma`
        is the scale parameter, and :math:`\alpha` is the skew parameter.

        """
        amp, loc, scale, alpha = params
        
        _x = alpha * (x - loc) / scale
        
        norm = np.sqrt(2 * np.pi * scale**2)**-1 * \
            np.exp(-(x - loc)**2 / (2 * scale**2))
            
        cdf = 0.5 * (1 + scipy.special.erf(_x / np.sqrt(2)))
        
        dist = amp * 2 * norm * cdf
        
        print('computed')
        return dist

    def _fit_skewnorms(self,
                       x,
                       *params):
        R"""
        Estimates the parameters of the distributions which consititute the 
        peaks in the chromatogram. 

        Parameters
        ----------
        x : `float`
            The time dimension of the skewnorm 
        params : list of length 4 x number of peaks, [amplitude, loc, scale, alpha]
            Parameters for the shape and scale parameters of the skewnorm 
            distribution. Must be provided in following order, repeating
            for each distribution.
                `amplitude` : float; > 0
                    Height of the peak.
                `loc` : float; > 0
                    The location parameter of the distribution.
                `scale` : float; > 0
                    The scale parameter of the distribution.
                `alpha` : float; > 
                    The skew shape parater of the distribution.

        Returns
        -------
        out : `float`
            The evaluated distribution at the given time x. This is the summed
            value for all distributions modeled to construct the peak in the 
            chromatogram.
        """
        if len(params) % 4 != 0:
            raise ValueError(
                "length of params must be divisible by 4\n"
                f"length of params = {len(params)}"
                )
        
        # Get the number of peaks and reshape for easy indexing
        n_peaks = int(len(params) / 4)
        params = np.reshape(params, (n_peaks, 4))
        out = 0

        # Evaluate each distribution
        for i in range(n_peaks):
            out += self._compute_skewnorm(x, *params[i])
        return out