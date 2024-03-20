import pandas as pd

df = pd.read_csv('Templates.csv',sep=';')

df.drop(['Category','Sub-Category'], axis=1, inplace=True)

for index,row in df.iterrows():
    charts = []
    row['Charts'] = row['Charts'][1:-1]
    aux = row['Charts'].split(',')
    for chart in aux:
        if chart == 'Nr records x nr variables':
            charts.append('nr_records_nr_variables')
        if chart == 'Correlation heatmap':
            charts.append('correlation_heatmap')
        #if chart == 'Histograms + MV':
        #    charts.append('histograms_numeric')
        #    charts.append('mv')
        #if chart == 'Scatter-plots':
        #    charts.append('scatter-plots')
        if chart == 'Histograms':
            charts.append('histograms_numeric')
        if chart == 'All_Histograms':
            #charts.append('histograms')
            charts.append('histograms_numeric')
            charts.append('histograms_symbolic')
        if chart == 'Boxplots':
            charts.append('boxplots')
        if chart == 'Single boxplots':
            charts.append('boxplots')
        if chart == 'Description':
            charts.append('description')
        if chart == 'Decision tree':
            charts.append('decision_tree')
        if chart == 'Class histogram':
            charts.append('class_histogram')
        if chart == 'Missing values':
            charts.append('mv')
        if chart == 'PCA':
            charts.append('pca')
        if chart == 'Overfitting Decision Tree':
            charts.append('overfitting_decision_tree')
        if chart == 'Overfitting Decision Tree + Overfitting KNN':
            charts.append('overfitting_decision_tree')
            charts.append('overfitting_knn')
        if chart == 'Overfitting KNN':
            charts.append('overfitting_knn')
        if chart == 'Overfitting RF':
            charts.append('overfitting_rf')
        if chart == 'Overfitting GB':
            charts.append('overfitting_gb')
        if chart == 'Overfitting MLP':
            charts.append('overfitting_mlp')
        if chart == 'Overfitting Accuracy + Recall':
            charts.append('overfitting_dt_acc_rec')
            
        df.at[index,'Charts'] = charts

df.to_csv('Templates1.csv', sep=";",index=False)