import libCB3 as cb
from WindPy import w
import pandas as pd
import numpy as np


def get_issueamount(codes):
    
    '''
    codes：list型，代码列表
    
    输出srsIssue
    '''
    
    if not w.isconnected(): w.start()
    
    strCode=','.join(codes)
    data=w.wss(strCode,'issue_amountact').Data[0]
    srsIssue = pd.Series(data, index = codes)
    
    return srsIssue


def getCredit(codes):
    strCodes = ','.join(codes)
    if not w.isconnected(): w.start()
    
    rt = w.wss(strCodes,"creditrating")
    srs = pd.Series(rt.Data[0], index=codes)
    return srs



def frameStrategy(obj, 
                  start='2017/12/29',
                  end=None,
                  defineMethod='default',
                  selMethod=None,
                  weightMethod='average',
                  roundMethod=21):
    
    obj.DB['Close'].fillna(method='pad',inplace=True)
    
    def getStartLoc(obj, date):

        if date in obj.DB['Amt'].index:
            
            i = obj.DB['Amt'].index.get_loc(date)
        else:
            fakeIndex = pd.to_datetime(obj.DB['Amt'].index)
            i = fakeIndex.get_loc(pd.to_datetime(date), method='ffill')
    
        return i
        

    def defineCodes(obj, method='default'):
        
        if method == 'default':
            
            return obj._excludeSpecial()
        
        elif method == 'nonEB':
            
            return obj._excludeSpecial(hasEB=0)
        
        elif type(method).__name__ == 'function':
            
            return method(obj)            
            
    
    def selectCodes(obj, codes, date, dfAssetBook, func=None):
        
        i = getStartLoc(obj, date)
        
        n = min([i,5])
        
        condition = (obj.DB['Amt'].iloc[i-n:i][codes].fillna(0).min() > 100000.0) & \
        (obj.DB['Outstanding'].iloc[i][codes] > 30000000.0)
        
        if func:
        
            tempCodes = list(condition[condition].index)
            moreCon = func(obj, codes, date, tempCodes, dfAssetBook)

            if not isinstance(moreCon, pd.Series):
                moreCon = pd.Series([True] * len(moreCon), index=moreCon)

            condition &= moreCon

        ## more codition: condition &= ...
        
        retCodes = list(condition[condition].index)
        
        if not retCodes:
            
            print('its a empty selection, when date: ',date)
        
        return retCodes
    
    
    def getWeight(obj, codes, date, method='average'):
        
        if method == 'average':
            
            ret = pd.Series(np.ones(len(codes)) / float(len(codes)), index=codes)
                        
            return ret
        
        elif method == 'fakeEv':
        
            srsIssue = get_issueamount(codes)
            srsFakeEv = obj.DB['Close'].loc[date, codes] * srsIssue
            
            return srsFakeEv / srsFakeEv.sum()
        
        elif method == 'fund':
            
            srsIssue = get_issueamount(codes)
            srsCredit = getCredit(codes)
            
            dictCreditWeighted = {"AAA":1.3,"AA+":1.15,"AA":1.0,"AA-":0.9,"A+":0.6}
            
            srsCredit = srsCredit.apply(lambda x: dictCreditWeighted[x] if x in list(dictCreditWeighted.keys()) else 0)
            print(srsCredit)
            
            srs = srsIssue * srsCredit
            print(srs / srs.sum())
            
            return srs / srs.sum()
            
        
        elif method == 'Ev':
        
            srsOutstanding = obj.DB['Outstanding'].loc[date, codes]
            srsEv = obj.DB['Close'].loc[date, codes] * srsOutstanding
            
            return srsEv / srsEv.sum()
        
        elif type(method).__name__ == 'function':
            
            return method(obj, codes, date)
            
    def roundOfAdjust(obj, start, method='daily'):
        
        i = getStartLoc(obj, start)
        
        if method == 'daily':
            
            return obj.DB['Amt'].index[i:]
        
        elif isinstance(method, int):
                                
            return obj.DB['Amt'].index[i:][::method]
    
    def checkBook(obj, dfRet, dfAssetBook, cash, date, cashRate = 0):
        
        if date == dfRet.index[0]:
            
            dfRet.loc[date]['NAV'] = 100
        
        else:
            i = dfRet.index.get_loc(date); j = obj.DB['Close'].index.get_loc(date)
            
            if len(dfAssetBook.index) == 1 and dfAssetBook.index[0] == 'Nothing':
                
                dfRet.iloc[i]['NAV'] = dfRet.iloc[i-1]['NAV'] * (1 + cashRate/252.0)
                cash *= 1 + cashRate/252.0
            else:
                
                codes = list(dfAssetBook.index)
                
                srsPct = obj.DB['Close'].iloc[j-1:j+1][codes].pct_change().iloc[-1] + 1.0
                
                cashW = 1 - dfAssetBook['w'].sum()
                
                t1 = (srsPct * dfAssetBook['costPrice'] * dfAssetBook['w']).sum() + cash * cashW * (1 + cashRate / 252.0) 
                t0 = (dfAssetBook['costPrice'] * dfAssetBook['w']).sum() + cash * cashW
                
                dfRet.iloc[i]['NAV'] = dfRet.iloc[i-1]['NAV'] * t1 / t0
                cash *= 1 + cashRate / 252.0
            
            
                
    intStart = getStartLoc(obj, start)
    if end is None:
        intEnd = -1
        dfRet = pd.DataFrame(index=obj.DB['Amt'].index[intStart:], columns=['NAV','LOG:SEL','LOG:WEIGHT'])
    else:
        intEnd = getStartLoc(obj, end)
        dfRet = pd.DataFrame(index=obj.DB['Amt'].index[intStart:intEnd+1], columns=['NAV','LOG:SEL','LOG:WEIGHT'])
    
    
    cash = 100.0
    
    codes = defineCodes(obj, defineMethod)
    isAdjustDate = roundOfAdjust(obj, start, roundMethod)
    
    dfAssetBook = pd.DataFrame(index=['Nothing'], columns=['costPrice', 'w'])
        
    for i,date in enumerate(dfRet.index):
        
        checkBook(obj, dfRet, dfAssetBook, cash, date)        
        
        if date in isAdjustDate:
            
            sel = selectCodes(obj, codes, date, dfAssetBook, selMethod)
            
            if sel:
                w = getWeight(obj, sel, date, weightMethod)
            else:
                sel = ['Nothing']
                w = 0.0
                
            dfAssetBook = pd.DataFrame(index=sel, columns=['costPrice', 'w'])
                
            dfAssetBook['costPrice'] = 100.0
            dfAssetBook['w'] = w
        
        dfRet['LOG:SEL'][date] = ','.join(list(dfAssetBook.index))
        dfRet['LOG:WEIGHT'][date] = ','.join([str(t) for t in list(dfAssetBook['w'])])
    
    return dfRet
