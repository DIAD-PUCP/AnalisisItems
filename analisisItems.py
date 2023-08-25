#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jinja2
from io import BytesIO
from zipfile import ZipFile
from collections import OrderedDict
from numpy.random import default_rng
from ctt_and_rasch import score, item_analysis, distractor_analysis, rasch

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

def all_A(columns):
    return pd.Series(['A']*len(columns),index=columns)

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

def raschWinsteps(X,key,anchors=None,suffix=''):
    jinja_env = jinja2.Environment(
        #donde están los templates, por defecto es la carpeta actual
        loader = jinja2.FileSystemLoader('.'),autoescape= True
    )
    tpl = jinja_env.get_template('tpl.con')
    confile = tpl.render(
        path='',
        key=''.join(key),
        anchors = anchors,
        names=X.columns,
        nitems=X.shape[1],
        suffix=suffix,
        data = X.fillna(' ').to_csv(None,header=False)
    )
    return confile

@st.cache_data
def analisisRasch(rsp,key,estructura,estructuraDTI):
    scored = score(rsp,key)
    dif = {}
    hab = {}
    graphs = {}
    confiles = {}
    estructuraDTI.index = scored.columns
    for c in estructura:
        dif[c],hab[c] = rasch(scored.filter(regex=f'{c}..'))
        
        b_ini = estructuraDTI.loc[estructuraDTI['comp']==c,'Medición'] #Dificultad de la estructura
        b_calc = dif[c]['measure'] #Dificultad calculada

        # Calcular los items a desanclar
        b_ini0 = b_ini - (b_ini[b_ini.notnull()].mean() - b_calc[b_ini.notnull()].mean())
        x = b_ini0[b_ini0.notnull()]
        y = b_calc[b_ini0.notnull()]
        keep = (np.abs(x-y) < 0.5).rename('anchored')
        color = np.where(keep,'blue','red')
        graphs[c] = scatter_plot(x,y,x.index,color)

        anclas = dif[c].join(keep)['anchored'] == True
        confiles[c] = raschWinsteps(rsp.filter(regex=f'{c}..'),key.filter(regex=f'{c}..'),anchors=np.where(anclas,b_ini,np.nan),suffix=c)
        diff = b_ini[anclas].mean() - b_calc[anclas].mean()
        st.write(f'Diff {c}: {diff}')
        #b_calc2 = np.where(anclas,b_ini,b_calc + diff)
        b_calc2 = b_calc + diff
        dif[c],hab[c] = rasch(scored.filter(regex=f'{c}..'),anchored_difficulty=b_calc2)
        dif[c]['measure'] = b_calc
        dif[c] = dif[c].join(anclas)
        dif[c]['measureA'] = b_calc2
        
    return dif,hab,graphs,confiles

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
        tabCTT, tabIRT, tabInsumos, tabDescargas = st.tabs(['CTT', 'IRT','Insumos','Descargas'])
        estDTI,est = leer_estructura(est_file)
        rsp = leer_respuestas(rsp_file,est)
        keys = leer_claves(key_file,rsp.columns)
        ctt_ia,ctt_da,scored = analisisCTT(rsp,keys,est)
        dif,hab,graphs,confiles= analisisRasch(rsp,keys,est,estDTI)
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
            comp = st.selectbox('Competencia',key='compirt',options=est.keys(),help='Seleccionar la competencia a analizar')
            st.subheader('Análisis IRT')
            if comp:
                st.pyplot(graphs[comp])
                st.dataframe(dif[comp],use_container_width=True)
                st.dataframe(hab[comp],use_container_width=True)

        with tabInsumos:
            st.subheader('Estructura')
            st.dataframe(estDTI)
            st.subheader('Respuestas')
            st.dataframe(rsp)
            st.subheader('Claves')
            st.dataframe(keys)

        with tabDescargas:
            l = []
            for k,v in dif.items():
                l.append(v)
            difRes = pd.concat(l)[['count','measureA','error','infit','outfit']].round(3)
            l = []
            for k,v in ctt_ia.items():
                l.append(v['items'])
            cttRes = pd.concat(l)[['mean','pbis']].round(3)
            difRes = pd.concat([difRes,cttRes],axis=1)
            st.write(difRes)
            st.download_button(
                'Descargar estructuras',
                data=difRes.to_csv(None),
                file_name="Estructura.csv"
            )

            tempZip = BytesIO()
            with ZipFile(tempZip,'w') as zf:
                for k,v in confiles.items():
                    with zf.open(f'{k}.con','w') as conBuffer:
                        conBuffer.write(v.encode('utf-8'))
            st.download_button(
                'Descargar archivos winsteps',
                data=tempZip.getvalue(),
                file_name='winsteps.zip',
                mime="application/zip"
            )
            tempZip.close()
main()