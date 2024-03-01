"""
2024-02-28 10:05:12 - Main
"""
from hplc_py.pipeline import pipeline
from hplc_py.pipeline import test_dataset_frames as test_df
def main():
    pline = pipeline.DeconvolutionPipeline()
    pline.run(
        data=test_df.ASSCHROM_DF,
        key_time=test_df.TIME_ASSCHROM,
        key_amp=test_df.AMP_ASSCHROM,
        )
    
if __name__ == "__main__":
    main()