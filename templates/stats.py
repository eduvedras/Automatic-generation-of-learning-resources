from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import gca,figure, savefig, show, subplots
from model.dslabs_functions import plot_bar_chart

file_tag = "OneImageDataset"

data = read_csv(file_tag + ".csv", sep=';')

#nr of records
print(f"Number of records: {len(data)}")

#distribution of charts
target = "Chart"

charts = ['boxplots','class_histogram','correlation_heatmap','decision_tree','histograms','histograms_numeric','mv','nr_records_nr_variables','overfitting_decision_tree','overfitting_dt_acc_rec','overfitting_gb','overfitting_knn','overfitting_mlp','overfitting_rf','pca','scatter-plots']

n_boxplots = 0
n_class_histogram = 0
n_correlation_heatmap = 0
n_decision_tree = 0
n_histograms = 0
n_histograms_numeric = 0
n_mv = 0
n_nr_records_nr_variables = 0
n_overfitting_decision_tree = 0
n_overfitting_dt_acc_rec = 0
n_overfitting_gb = 0
n_overfitting_knn = 0
n_overfitting_mlp = 0
n_overfitting_rf = 0
n_pca = 0
n_scatter_plots = 0

for index, row in data.iterrows():
    if 'boxplots' in row[target]:
        n_boxplots += 1
    elif 'class_histogram' in row[target]:
        n_class_histogram += 1
    elif 'correlation_heatmap' in row[target]:
        n_correlation_heatmap += 1
    elif 'overfitting_decision_tree' in row[target]:
        n_overfitting_decision_tree += 1
    elif 'decision_tree' in row[target]:
        n_decision_tree += 1
    elif 'histograms_numeric' in row[target]:
        n_histograms_numeric += 1
    elif 'histograms' in row[target]:
        n_histograms += 1
    elif 'mv' in row[target]:
        n_mv += 1
    elif 'nr_records_nr_variables' in row[target]:
        n_nr_records_nr_variables += 1
    elif 'overfitting_dt_acc_rec' in row[target]:
        n_overfitting_dt_acc_rec += 1
    elif 'overfitting_gb' in row[target]:
        n_overfitting_gb += 1
    elif 'overfitting_knn' in row[target]:
        n_overfitting_knn += 1
    elif 'overfitting_mlp' in row[target]:
        n_overfitting_mlp += 1
    elif 'overfitting_rf' in row[target]:
        n_overfitting_rf += 1
    elif 'pca' in row[target]:
        n_pca += 1
    elif 'scatter-plots' in row[target]:
        n_scatter_plots += 1

print(n_overfitting_decision_tree)
sum = n_boxplots + n_class_histogram + n_correlation_heatmap + n_decision_tree + n_histograms + n_histograms_numeric + n_mv + n_nr_records_nr_variables + n_overfitting_decision_tree + n_overfitting_dt_acc_rec + n_overfitting_gb + n_overfitting_knn + n_overfitting_mlp + n_overfitting_rf + n_pca + n_scatter_plots
values = [n_boxplots,n_class_histogram,n_correlation_heatmap,n_decision_tree,n_histograms,n_histograms_numeric,n_mv,n_nr_records_nr_variables,n_overfitting_decision_tree,n_overfitting_dt_acc_rec,n_overfitting_gb,n_overfitting_knn,n_overfitting_mlp,n_overfitting_rf,n_pca,n_scatter_plots]

print("SUM: " + str(sum))      
print(charts)
print("-------------")
print(values)

figure(figsize=(15, 6))
plot_bar_chart(
    charts,
    values,
    title=f"Data chart distribution",
)
savefig(f"statistics/{file_tag}_chart_distribution.png", bbox_inches='tight')
show()

templates = read_csv('Templates.csv', sep=';')
lst = []
dic = {}
for index, row in templates.iterrows():
    first_str = row['Template'].split('[')[0]
    lst_str = row['Template'].split(']')[-1]
    lst.append([first_str,lst_str, row['Category']])
    dic[row['Category']] = 0
    
categories = templates['Category'].unique()

for index, row in data.iterrows():
    for el in lst:
        if el[0] in row['Question'] and el[1] in row['Question']:
            dic[el[2]] += 1
            

print(dic)
values = []
cat = 0
while cat < len(categories):
    values.append(dic[categories[cat]])
    if type(categories[cat]) == float:
        categories[cat] = 'Without category'
    cat += 1

#categories.replace(None, 'None', inplace=True)
print(categories)
print(values)
figure(figsize=(10, 4))
plot_bar_chart(
    categories,
    values,
    title=f"Categories distribution",
)
savefig(f"statistics/{file_tag}_category_distribution.png", bbox_inches='tight')
show()