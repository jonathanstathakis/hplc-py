from scipy import signal
import seaborn as sns

print(sns.get_dataset_names())
for dset in sns.get_dataset_names():
    df = sns.load_dataset(dset)
    print(dset)
    print(df.head())

dowjones = sns.load_dataset("dowjones")
sns.lineplot(dowjones, x="Date", y="Price")
import matplotlib.pyplot as plt

plt.show()

p_idx, _ = signal.find_peaks(dowjones["Price"], prominence=0.01)
peak_widths, _, _, _ = signal.peak_widths(x=dowjones["Price"], peaks=p_idx)

print("#### PEAK WIDTHS ####")
print(peak_widths)
