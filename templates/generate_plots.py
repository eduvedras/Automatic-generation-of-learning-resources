from pandas import read_csv
import pandas as pd

file_tag = "Iris"
data = read_csv(file_tag + ".csv", index_col='Id', sep=',', decimal='.')

templates = read_csv('Templates.csv', sep=';')

print(templates['Charts'].unique())

from matplotlib.pyplot import figure, savefig, show
from dslabs_functions import plot_bar_chart

# Plot nr of records vs nr of variables
figure(figsize=(4, 2))
values: dict[str, int] = {"nr records": data.shape[0], "nr variables": data.shape[1]}
plot_bar_chart(
    list(values.keys()), list(values.values()), title="Nr of records vs nr variables"
)
savefig(f"images/{file_tag}_records_variables.png", bbox_inches='tight')
show()