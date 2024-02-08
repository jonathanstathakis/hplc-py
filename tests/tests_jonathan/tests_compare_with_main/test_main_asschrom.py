"""
Single test to produce a pickle of the main modules fit_peaks results for asschrom
"""
import hplc
def test_pk_main_chm_asschrom_fitted(
    main_chm_asschrom_fitted: hplc.quant.Chromatogram,
    main_chm_asschrom_fitted_pkpth: str,
):

    # raise RuntimeError("DEBUG MODE: NOT PICKLING")
    import pickle
    
    with open(main_chm_asschrom_fitted_pkpth, "wb") as f:
        pickle.dump(main_chm_asschrom_fitted, f)
        import warnings
        warnings.warn(f"main_chm_asschrom_fitted pickled at {main_chm_asschrom_fitted_pkpth}")
        