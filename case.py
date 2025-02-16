#%%
### Calling libraries
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from scipy.stats import linregress
from pmdarima import auto_arima

#%%
### Obtaining companies' data from SIDRA API
def api_data(url):    
    try:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        print('Request was successful')
    except requests.exceptions.RequestException as e:
        print(f"Error in requisition: {e}")
        exit()

    ### Transforming data into a DataFrame
    df = pd.DataFrame(data[1:])
    df = df[['D1N','D2N','V']]
    df[['D2N','V']] = df[['D2N','V']].apply(pd.to_numeric, errors='coerce')

    return df

#%%
### Opening population estimates data from xlsx file 
def xlsx_data(file, header):
    df = pd.read_excel(file, header=header)

    ### Filtering population data for ages 38-58 and UFs
    df = df[(df['IDADE']>=38) & (df['IDADE']<=58) & (~df['LOCAL'].isin(
            ['Brasil','Sul','Sudeste','Nordeste','Norte','Centro-Oeste'])) & (df['SEXO']=='Ambos')]
    
    ### Melting years columns to put it and estimated population each in a column
    years = df.loc[:, 2007:2020].columns
    df = df.melt(id_vars=['LOCAL','IDADE'], value_vars=years, var_name='ANO', value_name='POP_ESTIMADA')

    ### Grouping data by UF and year
    df = df.groupby(['LOCAL','ANO'])['POP_ESTIMADA'].sum().reset_index()
    
    return df

#%%
### Merging companies and population data
def df_merge(df_pop,df_comp):
    df = pd.merge(df_pop, df_comp, left_on=['LOCAL','ANO'], right_on=['D1N','D2N'])
    df.drop(['D1N','D2N'], axis=1, inplace=True)
    df.rename(columns={'V':'EMPRESAS_ATIVAS'}, inplace=True)
    ### Calculating ratio
    df['RAZAO'] = df['POP_ESTIMADA'] / df['EMPRESAS_ATIVAS']

    return df

#%%
### Using ADF test to verify if series are stationary or not
def adf_test(df):
    pred = []
    for uf in df['LOCAL'].unique():
        a = df[df['LOCAL']==uf].copy()
        adf = adfuller(a['RAZAO'])

        ### Applying Exponential Smoothing to stationary series and AUTOarima to non-stationary series
        if adf[1]<.05:
            print(f'{uf} - stationary series')
            model = ExponentialSmoothing(a['RAZAO'], trend='add', seasonal=None).fit()
        else:
            print(f'{uf} - non-stationary series')
            model = auto_arima(a['RAZAO'], seasonal=False, suppress_warnings=True)

        fc = model.forecast(2) if adf[1]<.05 else model.predict(n_periods=2)
        pred.append(pd.DataFrame({'LOCAL':uf, 'ANO':[2021,2022], 'RAZAO':fc}))

    pred = pd.concat(pred)
    df = pd.concat([df,pred]).reset_index(drop=True)

    return df

# #%%
# ### Using Exponential Smoothing as predict model for temporal series
# fc_r = []
# for uf in df['LOCAL'].unique():
#     y = df[df['LOCAL']==uf].copy()
#     model = ExponentialSmoothing(y['RAZAO'], trend='add', seasonal=None).fit()
#     fc = model.forecast(2)
#     fc_r.append(pd.DataFrame({'LOCAL':uf, 'ANO':[2021,2022], 'RAZAO':fc.values}))

# fc_r = pd.concat(fc_r)
# df = pd.concat([df,fc_r])

#%%
### Standardizing data to clustering
def clustering(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[['RAZAO']])

    ### Using KMeans to group temporal series and identify similar behaviors between UFs
    best_k, best_score = 2, -1
    for k in range(2,11): 
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(scaled_data)
        score = silhouette_score(scaled_data, labels) # Identifying better k to apply
        if score > best_score:
            best_k, best_score = k, score

    print(f'Melhor número de clusters: {best_k}')
    model = KMeans(n_clusters=best_k, random_state=42).fit(scaled_data)
    df['CLUSTER'] = model.labels_

    for c in range(best_k):
        uf_cluster = df[df['CLUSTER']==c]['LOCAL'].unique()
        print(f'Cluster {c}: {list(uf_cluster)}')

    return df

#%%
### Identifying similarity between UFs considering the clustering over all years
def uf_group(df):
    cluster_0=[]
    cluster_1=[]
    for uf in df['LOCAL'].unique():
        g = df[df['LOCAL']==uf].copy()
        m = g['CLUSTER'].mean()
        if m>.5:
            cluster_1.append(uf)
        else:
            cluster_0.append(uf)
    print('Similaridades entre estados:')
    print(f'Estados no cluster 0 - {set(cluster_0)}')
    print(f'Estados no cluster 1 - {set(cluster_1)}')

#%%
'''Identifying UFs more saturated and with more opportunities considering the clustering
    (mean for each UFs; if > 0.5 = 1, else = 0), the % of change on ratio, and the ratio mean
    of UF compared to the national; all metrics considered the last 4 years, including the
    predicted ones'''
def behavior(df):
    br_mean = df[df['ANO'].isin([2019,2020,2021,2022])].groupby(['ANO'])['RAZAO'].mean()

    df_sat = []
    sat = []
    opp = []
    for uf in df['LOCAL'].unique():
        z = df[df['LOCAL']==uf].copy()

        slope, intercept, r_value, p_value, std_err = linregress(z['ANO'], z['RAZAO'])

        z.loc[z['LOCAL']==uf,'TENDENCIA'] = slope
        tend = z['TENDENCIA'].iloc[-1]

        uf_mean = z[z['ANO'].isin([2019,2020,2021,2022])]['RAZAO'].mean()

        pred_g = z[z['ANO'].isin([2019,2020,2021,2022])]['RAZAO'].pct_change().sum()

        cluster_mean = z[z['ANO'].isin([2019,2020,2021,2022])]['CLUSTER'].mean()

        if cluster_mean<.5 and pred_g<0 and uf_mean<br_mean.mean():
            sat.append(uf)
        elif cluster_mean>.5 and pred_g>0 and uf_mean>br_mean.mean():
            opp.append(uf)

        df_sat.append({
            'ESTADO':uf,
            'TENDENCIA':tend,
            'MEDIA_RAZAO':uf_mean,
            'MEDIA_BR':br_mean.mean(),
            '%PREDICAO':pred_g,
            'CLUSTER': 1 if z['CLUSTER'].mean() > .5 else 0
        })

    df_sat = pd.DataFrame(df_sat)

    return df_sat, sat, opp

#%%
### Identifying tendencies of ratio over the years by UFs and showing the estimated prediction
def tend_plot(df):    
    palette = sns.color_palette("hsv", n_colors=len(df['LOCAL'].unique()))

    plt.figure(figsize=(12,6))

    for i,uf in enumerate(df['LOCAL'].unique()):

        x = df[(df['LOCAL']==uf)]
        plt.plot(x['ANO'], x['RAZAO'], label=uf, linestyle='-', alpha=.7, color=palette[i])

        p = x[(x['ANO']>2020)]
        if not p.empty:
            plt.plot(p['ANO'], p['RAZAO'], linestyle='dashed', alpha=.9, marker='o', color=palette[i])

    plt.xlabel('Ano')
    plt.ylabel('Razão Consumidores/Empresas')
    plt.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1,1.04))
    plt.grid()
    plt.show()

#%%
### Ploting clusters in 2D space by ratio and %growth ratio
def cluster_plot(df):
    df['RAZAO_%DIF'] = df.groupby('LOCAL')['RAZAO'].pct_change()

    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df, x='RAZAO', y='RAZAO_%DIF', hue='CLUSTER', palette='viridis')
    plt.xlabel('Razão Atual')
    plt.ylabel('Taxa de Crescimento da Razão (%)')
    plt.show()

#%%
### Boxplot to show how variables behaves between the clusters
def cluster_boxplot(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # 2x2 grid

    vars = ['POP_ESTIMADA', 'EMPRESAS_ATIVAS', 'RAZAO', 'RAZAO_%DIF']
    tit = ['População Estimada', 'Empresas Ativas', 'Razão', 'Razão % Diferença']

    for ax, var, titulo in zip(axes.flatten(), vars, tit):
        sns.boxplot(data=df, x='CLUSTER', y=var, ax=ax, palette='viridis')
        ax.set_title(titulo)
        ax.set_xlabel('Cluster')
        ax.set_ylabel(None)

    plt.tight_layout()
    plt.show()

#%%
### Plot showing UFs with cummulative % of change in ratio in the last 4 years
def ratio_plot(df_sat):
    df_sat_sorted = df_sat.sort_values(by='%PREDICAO', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='ESTADO', y='%PREDICAO', data=df_sat_sorted, palette='coolwarm')
    plt.xticks(rotation=90)
    plt.xlabel(None)
    plt.ylabel('Crescimento/Redução (%)')
    plt.show()
#%%
###------------------------ Main program ------------------------###
url = 'https://apisidra.ibge.gov.br/values/t/1757/n3/all/p/2007-2020/v/410/c319/104030/f/u?formato=json'
comp = api_data(url)

file = 'projecoes_2024_tab1_idade_simples.xlsx'
pop = xlsx_data(file, header=5)

df = df_merge(pop,comp)

df = adf_test(df)

df = clustering(df)

df_beh, saturat_uf, opport_uf = behavior(df)

### UFs more related by clustering
uf_group(df)

### UFs with most opportunities and the more saturated obtained by the combination of mean 
### clustering, %change ratio, and the ratio mean of UF compared to the national, considering the last 
### 4 years

print('Estados com maiores oportunidades futura:')
print(opport_uf)
print('Estados mais saturados:')
print(saturat_uf)

### Tendencies of the ratio over the years with marked model estimated years (2021-2022)
tend_plot(df)

### Clusters plotted by ratio and %chance ratio; clusters 0 have minor ratio, while clusters 1 have major 
cluster_plot(df)

### Boxplot of the variables grouped by each cluster to observe behavior of them
cluster_boxplot(df)

### Plot with %change ratio in the last 4 years, showing UFs with tendency to be more saturated, with more opportunities or neutral
ratio_plot(df_beh)

#%%