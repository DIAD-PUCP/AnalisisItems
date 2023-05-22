#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from ctt_and_rasch import score, item_analysis, distractor_analysis, rasch, raschWinsteps


def leer_estructura(archivo,col_comp='Competencia'):
    comp = {
        '011':'MAT',
        '014':'RED',
        '013':'LEC'
    }
    dti = pd.read_excel(archivo,skiprows=2,dtype={'Competencia':str,'Tema':str,'SubTema':str,'Categoria':str})
    dti['comp'] = dti[col_comp].apply(lambda x: comp[x])
    nitems = dti['comp'].value_counts().rename('nitems').to_dict()
    est = OrderedDict()
    for v in dti['comp'].unique():
        est[v] = nitems[v]
    return dti,est

def leer_respuestas(archivo,estructura,idlen=5):
    l = [(0,idlen)]
    start = idlen
    for v,numchars in estructura.items():
        l.append((start,start:=start+numchars))
    cols = ['EXAMEN'] + [k for k in estructura.keys()]
    df = pd.read_fwf(
        archivo,
        header=None,
        dtype=str,
        encoding='iso-8859-1',
        names= ['line'],
        widths= [start]
    )
    for (start,end),name in zip(l,cols):
        df[name] = df['line'].str[start:end]
    for col in cols[1:]:
        d = df[col].str.split(
            '',expand=True
        ).iloc[:,1:-1].rename(
            columns = lambda x: f'{col}{str(x).zfill(2)}'
        )
        df = df.drop(columns=[col]).join(d)
    return df.drop(columns=['line']).set_index('EXAMEN')

def leer_claves(path,colnames):
    clave = pd.read_excel(path).T
    clave.columns = colnames
    return clave.iloc[0]

@st.cache_data
def analisisCTT(rsp,keys,est):
    scored = score(rsp,keys)
    ia = {}
    da = {}
    for v in est:
        ia[v] = item_analysis(scored.filter(regex=f'{v}..'))
        da[v] = distractor_analysis(rsp.filter(regex=f'{v}..'),keys.filter(regex=f'{v}..'))
    return ia,da,scored

def scatter_plot(x,y,labels,colors,cutoff=0.5):
    fig,ax = plt.subplots(figsize=(8,5))
    xvals = np.arange(np.min(x),np.max(x)+0.2,0.1)
    ax.set_xlabel("Dificultad estructura")
    ax.set_ylabel("Dificultad calulada")
    ax.scatter(x=x,y=y,c=colors,s=8)
    for xi,yi,text in zip(x,y,labels):
        ax.annotate(text=text,xy=(xi,yi),size=8)
    ax.plot(xvals,xvals,'-',color="gray")
    ax.fill_between(xvals,xvals - cutoff,xvals + cutoff,alpha=0.2)
    return fig

@st.cache_data
def analisisRasch(scored,est,estDTI):
    dif = {}
    hab = {}
    graphs = {}
    estDTI.index = scored.columns
    for v in est:
        dif[v],hab[v] = rasch(scored.filter(regex=f'{v}..'))
        raschWinsteps(scored.filter(regex=f'{v}..'))
        b_calc = dif[v]['measure']
        b_ini = estDTI.loc[estDTI['comp']==v,'Medición']
        b_ini0 = b_ini - (b_ini[b_ini.notnull()].mean() - b_calc[b_ini.notnull()].mean())
        
        x = b_ini0[b_ini0.notnull()]
        y = b_calc[b_ini0.notnull()]
        
        keep = (np.abs(x-y) < 0.5).rename('keep')
        color = np.where(keep,"blue","red")

        # restar la media de los que queden anclados
        b_ini0 = pd.concat([b_ini0,keep],axis=1)
        diff_index = b_ini0['Medición'].notnull() & b_ini0['keep']
        diff = b_ini0.loc[diff_index,'Medición'].mean() - b_calc[diff_index].mean()
        b_calc = b_calc + diff + (b_ini[diff_index].mean()) - b_calc[diff_index].mean()
        #st.write(b_calc)
        
        graphs[v] = scatter_plot(x,y,x.index,color)
    return dif,hab,graphs

def main():
    if 'processed' not in st.session_state:
        st.session_state['processed'] = False
    st.title('Análisis de ítems')

    with st.sidebar:
        with st.form('insumos'):
            st.subheader('Insumos')
            est_file = st.file_uploader('Archivo de estructura',help='Es el mismo archivo de estructura que se sube a DTI')
            rsp_file = st.file_uploader('Archivo de respuestas',help='El archivo de respuestas (debe contener solo una versión)')
            key_file = st.file_uploader('Archivo de claves',help='Archivo de claves exportado en diagramación')
            procesar = st.form_submit_button('PROCESAR')

    if st.session_state['processed'] or procesar:
        st.session_state['processed'] = True
        tabCTT, tabIRT, tabInsumos = st.tabs(['CTT', 'IRT','Insumos'])
        estDTI,est = leer_estructura(est_file)
        rsp = leer_respuestas(rsp_file,est)
        keys = leer_claves(key_file,rsp.columns)
        ctt_ia,ctt_da,scored = analisisCTT(rsp,keys,est)
        dif,hab,graphs= analisisRasch(scored,est,estDTI)
        with tabCTT:
            st.subheader('Análisis CTT')
            st.dataframe(pd.DataFrame(ctt_ia).T.drop(columns='items'))
            comp = st.selectbox('Competencia',options=est.keys(),help='Seleccionar la competencia a analizar')
            if comp:
                items = ctt_ia[comp]['items']
                st.dataframe(items,use_container_width=True)
                item = st.selectbox('Item',options=items.index,help='Seleccionar el ítem a analizar')
                if item:
                    da = ctt_da[comp]
                    st.dataframe(da.loc[(item,),:],use_container_width=True)
        with tabIRT:
            comp = st.selectbox('Competencia',key="compirt",options=est.keys(),help='Seleccionar la competencia a analizar')
            st.subheader('Análisis IRT')
            if comp:
                st.pyplot(graphs[comp])
                st.dataframe(dif[comp],use_container_width=True)
                st.dataframe(dif[comp].describe(),use_container_width=True)

        with tabInsumos:
            st.subheader('Estructura')
            st.dataframe(estDTI)
            st.subheader('Respuestas')
            st.dataframe(rsp)
            st.subheader('Claves')
            st.dataframe(keys)

main()