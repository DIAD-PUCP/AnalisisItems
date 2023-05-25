#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize

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

def rasch_estimate_cmle(X,max_iter=10000,epsilon=0.0001):
    nperson,nitem = X.shape
    difficulty = np.random.randn(nitem)
    ability = np.zeros(nperson)

    def loglike(difficulty):
        difficulty = difficulty - np.mean(difficulty)
        success = np.exp(-difficulty)/(1+np.exp(-difficulty))
        likelihood = X * success + (1-X)*(1-success) 
        return -np.sum(np.sum(likelihood))
    
    result = minimize(loglike, difficulty, method='BFGS')
    difficulty = result.x
    return difficulty,ability



def rasch_estimate_jmle(X,max_iter=10000,epsilon=0.0001):
    item = np.nanmean(X,axis=0) # proportion of correct per item
    difficulty = np.log((1-item)/item) # log odds of failure
    difficulty = difficulty- np.nanmean(difficulty) # Ajust mean difficulty to 0
    
    person = np.nanmean(X,axis=1) # proportion of correct per person
    ability = np.log(person/(1-person)) # log odds of success
    
    i = 0
    sumsq_res = np.inf
    while (i< max_iter and sumsq_res > epsilon):
        # Calculate expected values
        dif,ab = np.meshgrid(difficulty,ability)
        expected =  np.exp(ab-dif)/(1+np.exp(ab-dif)) # P(X=1)

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
    kurtosis = np.multiply(variances,expected**3 + (1-expected)**3)
    return difficulty,ability,expected,variances,kurtosis

def fit_stats(X,expected,variances,kurtosis,axis):
    residuals = X - expected
    fit = (residuals**2)/variances

    outfit = np.mean(fit,axis=axis)
    var_out = np.mean((kurtosis/(variances**2))-1,axis=axis)/kurtosis.shape[axis]
    zstdoutfit = (np.cbrt(outfit)-1)*(3/np.sqrt(var_out)) + (np.sqrt(var_out)/3)

    infit = np.sum(residuals**2,axis=axis)/np.sum(variances,axis=axis)
    var_in = np.sum((kurtosis-(variances**2)),axis=axis)/(np.sum(variances,axis=axis)**2)
    zstdinfit = (np.cbrt(infit)-1)*(3/np.sqrt(var_in)) + (np.sqrt(var_in)/3)

    return pd.DataFrame({'infit':infit,'zinfit':zstdinfit,'outfit':outfit,'zoutfit':zstdoutfit})

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
    difficulty,ability,expected,variances,kurtosis = rasch_estimate_jmle(x)
    #print(rasch_estimate_cmle(x))
    
    # Calculate error
    dif_error = np.sqrt(1/np.sum(variances,axis=0))
    ab_error = np.sqrt(1/np.sum(variances,axis=1))
    
    item_fit = fit_stats(x,expected,variances,kurtosis,axis=0)
    person_fit = fit_stats(x,expected,variances,kurtosis,axis=1)

    dif = pd.DataFrame({'measure':difficulty,'error':dif_error},index=x.columns).join(item_fit)
    ab = pd.DataFrame({'measure':ability,'error':ab_error},index=x.index).join(person_fit)

    return resDif.join(dif),resAb.join(ab)