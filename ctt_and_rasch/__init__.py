#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import scipy.stats as stats
import jinja2
import os

def score(rsp,clave):
    return (rsp == clave).astype(int)

def polyserial(x,y):
    z = stats.zscore(x)
    N = len(y)
    y = pd.Categorical(y).codes + 1
    _,tau = np.unique(y,return_counts=True)
    tau = stats.norm.ppf(np.cumsum(tau[:-1])/N)
    xy = stats.pearsonr(x,y)[0]
    pop = np.std(y,ddof=1)*np.sqrt((N-1)/N)
    return (xy * pop)/np.sum(stats.norm.pdf(tau))

def item_analysis(df):
    nperson,nitem = df.shape
    scores = df.sum(axis=1)
    mean = scores.mean()
    sd = scores.std()
    item_var = df.var()
    alpha = (nitem/(nitem-1))*(1-item_var.sum()/scores.var())
    pvalues = df.mean()
    def item_stats(x):
        scored_deleted = df.drop(columns=x.name).sum(axis=1)
        pbis = df[x.name].corr(scored_deleted)
        alphad = ((nitem-1)/(nitem-2))*(1-item_var.drop(x.name).sum()/scored_deleted.var())
        bis = scored_deleted.corr(df[x.name],method=polyserial)
        return pbis,bis,alphad
    items = pd.concat([pvalues,df.apply(item_stats).T],axis=1)
    items.columns = ["mean","pbis","bis","alphaIfDel"]
    return {"nperson":nperson,"nitem":nitem,"mean":mean,"sd":sd,"alpha":alpha,"items":items}

def distractor_analysis(df,key):
    scored = (df == key).astype(int)
    scores = scored.sum(axis=1).rename('score')
    q = pd.qcut(scores,q=4,labels=['lower','50%','75%','upper']).rename('q')
    def item_distractor_analysis(x):
        dummys = pd.get_dummies(x)
        n = dummys.sum().rename('n')
        p = dummys.mean().rename('p')
        pbis = dummys.apply(lambda y: y.corr(scores-y)).rename('pbis')
        pq = dummys.join(q).groupby('q').mean().T
        disc = (pq['upper'] - pq['lower']).rename('disc')
        res = pd.concat([n,p,pbis,disc,pq],axis=1).rename_axis('choice')
        res['correct'] = res.index == key[x.name]
        res = res.unstack()
        return res
    res = df.apply(item_distractor_analysis).T.stack()
    res['n'] = res['n'].astype(int)
    return res.reindex(columns=['correct','n','p','pbis','disc','lower','50%','75%','upper'])

def rasch_estimate(X,epsilon=0.001,max_iter=10000):
    item = np.nanmean(X,axis=0) # proportion of correct per item
    difficulty = np.log((1-item)/item) # log odds of failure
    difficulty = difficulty- np.nanmean(difficulty) # Ajust mean difficulty to 0
    
    person = np.nanmean(X,axis=1) # proportion of correct per person
    ability = np.log(person/(1-person)) # log odds of success
    
    i = 0
    sumsq_res = np.inf
    while (i< max_iter and sumsq_res > epsilon):
        # Calculate expected values
        dif,hab = np.meshgrid(difficulty,ability)
        expected =  np.exp(hab-dif)/(1+np.exp(hab-dif)) # P(X=1)

        # Variances
        variances = np.multiply(expected , 1-expected)# p(1-p)
        dif_var = np.sum(-variances,axis=0)
        ab_var = np.sum(-variances,axis=1)

        # Residuals
        res = X - expected
        dif_res = -np.nansum(res,axis=0)
        ab_res = np.nansum(res,axis=1)
        
        # New difficulties and habilities
        difficulty = difficulty - (dif_res/dif_var)
        difficulty = difficulty - np.mean(difficulty) # Ajust mean difficulty to 0

        ability = ability - (ab_res/ab_var)
        
        sumsq_res = np.sum(ab_res**2)
        i = i+1
    return difficulty,ability,expected,variances

def rasch_fit(X,expected,variances):
    #Kurtosis
    C = np.multiply(expected**4,1-expected) + np.multiply(expected,(1-expected)**4)

    #Residuals
    res = X - expected
    
    #Fit values
    fit = (res**2)/variances

    #OutFit
    C = np.multiply(expected**4,1-expected) + np.multiply(expected,(1-expected)**4)
    dif_outfit = np.mean(fit,axis=0)
    dif_varoutfit2 = np.mean(fit**2,axis=0) - dif_outfit**2
    dif_varoutfit3 = np.var(fit,axis=0)
    dif_varoutfit = np.sum((C/(variances**2))/(C.shape[0]**2),axis=0) - (1/C.shape[0])
    ab_outfit = np.mean(fit,axis=1)
    dif_zstdoutfit = (np.cbrt(dif_outfit)-1)*(3/np.sqrt(dif_varoutfit)) + (np.sqrt(dif_varoutfit)/3)
    

    #InFit
    dif_infit = np.sum(res**2,axis=0)/np.sum(variances,axis=0)
    dif_varinfit = np.sum((C-(variances**2)),axis=0)/(np.sum(variances,axis=0)**2)
    ab_infit = np.sum(res**2,axis=1)/np.sum(variances,axis=1)
    dif_zstdinfit = (np.cbrt(dif_infit)-1)*(3/np.sqrt(dif_varinfit)) + (np.sqrt(dif_varinfit)/3)
    
    return dif_outfit,dif_zstdoutfit,ab_outfit,dif_infit,dif_zstdinfit,ab_infit

def rasch(X):
    resDif = pd.DataFrame(index=X.columns)
    resAb = pd.DataFrame(index=X.index)

    resDif = pd.concat([resDif,X.sum().rename('score')],axis=1)
    resDif["count"] = X.shape[0]

    resAb = pd.concat([resAb,X.sum(axis=1).rename('score')],axis=1)
    resAb["count"] = X.shape[1]

    maxAb = X.mean(axis=1)==1
    minAb = X.mean(axis=1)==0

    maxDif = X.mean()==1
    minDif = X.mean()==0
    x = X.loc[~maxAb & ~minAb, ~maxDif & ~minDif]
    difficulty,ability,expected,variances = rasch_estimate(x)
    
    # Calculate error
    dif_error = np.sqrt(1/np.sum(variances,axis=0))
    ab_error = np.sqrt(1/np.sum(variances,axis=1))
    
    dif_outfit,dif_zstdoutfit,ab_outfit,dif_infit,dif_zstdinfit,ab_infit = rasch_fit(x,expected,variances)
    
    dif = pd.DataFrame({'measure':difficulty,'error':dif_error,'infit':dif_infit,'zstdinfit':dif_zstdinfit,'outfit':dif_outfit,'zstdoutfit':dif_zstdoutfit},index=x.columns)
    ab = pd.DataFrame({'measure':ability,'error':ab_error,'infit':ab_infit,'outfit':ab_outfit},index=x.index)

    return resDif.join(dif),resAb.join(ab)

def raschWinsteps(X):
    jinja_env = jinja2.Environment(
        #donde están los templates, por defecto es la carpeta actual
        loader = jinja2.FileSystemLoader('.'),autoescape= True
    )
    tpl = jinja_env.get_template('tpl.con')
    X.to_csv('data.csv',header=False)
    confile = tpl.render(path=os.getcwd(),key='1'*X.shape[1],names=X.columns,nitems=X.shape[1])
    with open('con.con','w') as f:
        f.write(confile)
    print(f'flatpak run --command=bottles-cli com.usebottles.bottles run -p Winsteps -b Winsteps -- batch=yes "Z:{os.getcwd()}/test.con" "Z:{os.getcwd()}/out.log"')