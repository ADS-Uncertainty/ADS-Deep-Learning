import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os

for i in range(10):
    # Replace it with your file path
    array2 = np.load('/NEW_Per20/DEs_L_' + str(i + 1) + '.npy')
    array1 = np.load('/NEW_Per20/MCD_L_' + str(i + 1) + '.npy')

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['axes.labelsize'] = 15.5
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 15.5
    plt.rcParams['ytick.labelsize'] = 15.5
    plt.rcParams['legend.fontsize'] = 15.5

    std_dev1 = np.std(array1)
    std_dev2 = np.std(array2)

    print(std_dev1)

    green_color = '#48B348'
    blue_color = '#246CE6'

    plt.figure(figsize=(6, 6))

    bins = np.linspace(min(array1.min(), array2.min()), max(array1.max(), array2.max()), 12)
    bin_width = bins[1] - bins[0]
    offset = bin_width * 0.33

    plt.hist(array1, bins=(bins - offset / 2) + 25000, width=bin_width * 0.33, label='MCD', color=green_color,
             edgecolor='black', align='mid')
    plt.hist(array2, bins=(bins + offset / 2) + 25000, width=bin_width * 0.33, label='DE', color=blue_color,
             edgecolor='black', align='mid')

    def scientific_notation(x, pos):
        if x == 0:
            return "0"
        try:
            exponent = int(np.log10(x))
            coefficient = int(x / (10 ** exponent))
            # Unicode 超级指数
            superscripts = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸',
                            '9': '⁹'}
            exponent_str = ''.join(superscripts.get(digit, '') for digit in str(exponent))
            return f'{coefficient}×10{exponent_str}'
        except (ValueError, OverflowError):
            return ""


    formatter = FuncFormatter(scientific_notation)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.ylim(0, 16000)  # Modify to the scope you need
    plt.xlim(-15000, 560000)  # Modify to the scope you need
    plt.title('Top 20%-Level-' + str(i + 1))
    plt.xlabel('Trajectory ID')
    plt.ylabel('Count')
    plt.tick_params(axis='y', which='both', direction='in')
    plt.yticks(rotation=90)
    plt.yticks(np.arange(0, plt.ylim()[1] + 1, 2000))
    plt.tick_params(axis='x', which='both', direction='in')
    # plt.legend(ncol=2)
    plt.legend(
        ncol=2,
        fontsize=15.5,
        bbox_to_anchor=(1, 1),
        handletextpad=0.2,
        borderpad=0.2,
        labelspacing=0.2
    )

    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.15, left=0.15, right=0.85, top=0.85)
    # Replace it with your target file path
    plt.savefig('/Top20/Top 20%-Level-' + str(i + 1) + '.png')  # save as png
    # plt.show()