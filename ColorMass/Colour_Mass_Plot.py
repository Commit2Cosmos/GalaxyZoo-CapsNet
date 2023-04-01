import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

color_mass_labels = np.load(f'./ColorMass/color_mass_labels.npy')

# print(color_mass_labels)

sns_pd = pd.DataFrame(color_mass_labels, columns=['x', 'y'])

print(sns_pd)

fig, ax = plt.subplots(figsize=(15, 12))
# Basic 2D density plot
# sns.kdeplot(data=merged, x="MPAJHU_MEDIAN_MASS", y="SDSS_petromag_abs_u_kcorr_dustcorr", cmap ='Greens')
#sns.kdeplot(data=csv, x="LOG_MSTELLAR", y="U-R")

#* Add in the two linear green valley lines.
x1 = np.linspace(9, 11.5)
y1 = 0.25*x1 - 0.24
sns.lineplot(x=x1, y=y1, ax=ax, color='green')

x2 = np.linspace(9, 11.5)
y2 = 0.25*x2 - 0.75
sns.lineplot(x=x2, y=y2, ax=ax, color='green')


#* Create the Gaussian KDE plot using seaborn

sns.kdeplot(data=sns_pd, x="y", y="x", levels=6, thresh=0.2, color='black', ax=ax)


#* Format the plot
plt.subplots_adjust(bottom=0.19)
plt.xlabel("Stellar Mass (Log($M_{*}$/$M_{\odot}$))", fontsize=15)
plt.ylabel("u-r colour (dust and k-corrected)", fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
ax.set_ylim(0.8, 3.25)
ax.set_xlim(8.75, 11.75)
plt.show()