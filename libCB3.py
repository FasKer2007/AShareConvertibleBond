# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:50:13 2015

@author: yangbing
"""
import datetime as dt
import pandas as pd
import numpy as np
from WindPy import w
#特殊品种，计算指数时要剔除，数据库里暂且保留，但基本已经没有意义的品种，多为早年条款特殊或者因股权分置改革而退市的品种

lstSpecial=['100016.SH','100096.SH','100567.SH','110009.SH',\
'110010.SH','110036.SH','110037.SH','110219.SH','110232.SH',\
'110317.SH','110325.SH','110398.SH','110418.SH','110598.SH',\
'125024.SZ','125528.SZ','125629.SZ','125630.SZ','125729.SZ',\
'125822.SZ','125898.SZ','125930.SZ','125937.SZ','125959.SZ',\
'125302.SZ','125960.SZ','125302.SZ','110001.SH']

def _windChecker():
    if not w.isconnected(): w.start()

	
def str2dt(strDate):
    '''早年缩短代码用的'''
    return pd.to_datetime(strDate)


def dt2str(d,sep='/'):
    '''早年缩短代码用的'''
    return dt.strftime(d, sep.join(["%Y","%m","%d"]))
    
	
def myTDays(start, end):
    '''调万得取start, end之间的交易日'''
    _windChecker()
    
    dates = pd.to_datetime(w.tdays(start, end).Data[0]).apply(lambda x: dt2str)
	
    return dates
        

def cicc_read_from_wind(code, field, start_date, end_date, *others):
    # 早年python和wind功能不全时所写，可废止
    '''
    输入code，field、起止日期后直接输出pd.DataFrame型数据
    *others可以输入一些特殊的选项，比如impliedVol是"rfIndex=1"
    '''
    
    _windChecker()

    strCodes = ','.join(code)    
    _, df = w.wsd(strCodes, field, start_date, end_date, others, usedf=True)

    return df

def tblUpdate(csvName, toDate, field, kwargs, method = 'api'):
    
    '''
    更新表格的基本公式，这里要求field本来就是wind api或者rs。fieldReflect定义过的
    csvName：待更新表的csv文件名
    toDate：截止日
    field：字面意思
    kwargs：不行就''
    method：可以是api也可以是sql，取决于哪个方便
    '''
    # 读原表，并看看最后一次更新是什么日子    
    df = pd.read_csv(csvName, index_col=0)
    lastDayInDf = str(df.index[-1])
    # 取最后一天开始至今的交易日，若不大于1则停止，否则更新
    _windChecker()
    
    dates = w.tdays(lastDayInDf, toDate).Data[0]
    
    if len(dates) > 1:
        
        dates = [dt2str(dates[i]) for i in range(1,len(dates))]
        
    else:
        
        print('No Need to update' + csvName)
        return df
    # 更新进行时，更新好了挂在df后面
    if method == 'api' or field == 'impliedvol':
        
        temp = cicc_read_from_wind(df.columns, field, dates[0], dates[-1], kwargs)       
        
        df = df.append(temp)

    else:

        temp = rs.sqlFetching(list(df.columns), dates[0], dates[-1], field)
        
        temp = temp.reindex(dates)
    
        if field == 'amt': #因为sql库是千元为单位的
            
            temp *= 1000.0
            df = df.append(temp)
        
        elif field == 'clause_conversion2_bondlot': #因为sql库是亿元为单位的
            
            temp *= 100000000.0
            df = df.append(temp)
            df = df.fillna(method='pad')
    
        else:
            
            df = df.append(temp)
    
    return df        
    
def get_issueamount(codes: List) -> pd.Series:
    # 取发行额
    _windChecker()
    strCode=','.join(codes)
    data=w.wss(strCode,'issue_amountact').Data[0]
    srsIssue = pd.Series(data, index = codes)
    
    return srsIssue

    
class cb_data(object):
    
    dictData = {'Close':{'file':'tClose.csv','field':'close','kwarg':'','f':np.mean},
            'ConvV':{'file':'tConvValue.csv','field':'convvalue','kwarg':'','f':np.mean},
            'Strb' :{'file':'tStrbValue.csv','field':'strbvalue','kwarg':'','f':np.mean},
            'ConvPrem':{'file':'tConvPremiumRatio.csv','field':'convpremiumratio',
            'kwarg':'','f':np.mean},
            'StrbPrem':{'file':'tStrbPremiumRatio.csv','field':'strbpremiumratio',
            'kwarg':'','f':np.mean},
            'Outstanding':{'file':'tOutstanding.csv','field':'clause_conversion2_bondlot',
            'kwarg':'','f':np.sum},
            'YTM':{'file':'tYtm.csv','field':'ytm_cb','kwarg':'','f':np.mean},
            'ImpliedVol':{'file':'tImplVol.csv','field':'impliedvol','kwarg':'rfIndex=1','f':np.mean},
            'Amt':{'file':'tAmt.csv','field':'amt','kwarg':'','f':np.sum},
            'Ptm':{'file':'tPtm.csv','field':'ptmyear','kwarg':'','f':np.mean}}

    lstSpecial=['100016.SH','100096.SH','100567.SH','110009.SH','110010.SH',\
        '110036.SH','110037.SH','110219.SH','110232.SH','110317.SH',\
        '110325.SH','110398.SH','110418.SH','110598.SH','125024.SZ',\
        '125528.SZ','125629.SZ','125630.SZ','125729.SZ','125822.SZ',\
        '125898.SZ','125930.SZ','125937.SZ','125959.SZ','125960.SZ','125302.SZ']


    def __init__(self):
        
        self.DB = {}
        for key,value in list(cb_data.dictData.items()):
            self.DB[key] = pd.read_csv(value['file'], index_col = 0)
        self.date = self.DB["Amt"].index[-1]
        self.codes = self.selByAmt(self.date)
        
        
        
    def update(self, strToday, method='api'):
        
        for key,value in list(cb_data.dictData.items()):
            
            csvName = value['file']
            field = value['field']
            kwarg = value['kwarg']            
            self.DB[key] = tblUpdate(csvName, strToday, field, kwarg, method)
            print(key + ' has updated to ' + strToday)
        
    
    def outPut(self, pre_fix=""):        
        
        for key,value in list(cb_data.dictData.items()):
            self.DB[key].to_csv(pre_fix + value['file'])                            


    def summary(self):

        _windChecker()
        mat01 = self.DB["Amt"].applymap(lambda x: 1 if x > 0 else 0)
        dfRet = pd.DataFrame(index=mat01.index)
        for key, df in self.DB.items():
            if key in cb_data.dictData:
                f = cb_data.dictData[key]['f']
                srs = (df * mat01).apply(f, axis=1)

                dfRet[key] = srs
        self.Indicator = dfRet
        return dfRet
    
    def selByAmt(self, date, other=None):
        t = self.DB['Amt'].loc[date] > 0
        
        if other:
            for k, v in other.items():
                t *= self.DB[k].loc[date].between(v[0], v[1])
                
        codes = list(t[t].index)
        return codes
    
    def selByAmtPeriod(self, start, end):
        
        t = self.DB['Amt'].loc[start:end].sum() > 0
        return list(t[t].index)
        

    def _excludeSpecial(self, hasEB=1):
        
        columns =  set(list(self.DB['Amt'].columns))
        columns -= set(cb_data.lstSpecial)
        columns =  list(columns)        
        
        if not hasEB:
            
            for code in columns:
                
                if code[:3] == '132' or code[:3] == '120':
                    columns.remove(code)
        
        return columns    
    
    def insertNewKey(self, new_codes, method='api'):
        
        for key,value in list(self.DB.items()):
            
            print(key)
            diff = list(set(new_codes) - set(value.columns))
												
            if diff:

                if not key in ['StockPctChg','QConvV']:
                    		    
                    field = cb_data.dictData[key]['field']
                    kwarg = cb_data.dictData[key]['kwarg']
        
                    start = self.DB[key].index[0] ; end = self.DB[key].index[-1]
                    
                    if method == 'api' or field == 'impliedvol':
                    
                        df = cicc_read_from_wind(diff, field, start, end, kwarg)
                    
                    else:
                        
                        df = rs.sqlFetching(diff, start, end, field)
                        dates = myTDays(start, end)

                        df = df.reindex(dates)
                        
                        if field == 'amt':
                            
                            df *= 1000.0
                            
                        elif field == 'clause_conversion2_bondlot':
                            
                            df *= 100000000.0
                            df = df.fillna(method='pad')
                        
                    value = value.join(df)

                    self.DB[key] = value
            else:
                
                print('Key No Need to update!')

    def display(self, outPut=0, field=None, date=None):

        if not date:
            date = self.DB['Amt'].index[-1]

        if not field:
            field = list(cb_data.dictData.keys())

        t = self.DB['Amt'].loc[date] > 0
        lstCode = list(t[t].index)
        
        dfRet = pd.DataFrame(index=lstCode, columns=field)

        for f in field:
            for code in dfRet.index:
                dfRet.loc[code, f] = self.DB[f].loc[date, code]        

        if outPut:
            
            print(dfRet.head(20))

        return dfRet
            
                                              
def getStartFake(obj, date):

    if date in obj.DB['Amt'].index:
        
        i = obj.DB['Amt'].index.get_loc(date)
    else:
        fakeIndex = obj.DB['Amt'].index.map(str2dt)
        i = fakeIndex.get_loc(str2dt(date), method='ffill')

    return i, obj.DB['Amt'].index[i]

def getCredit(codes):
    strCodes = ','.join(codes)
    if not w.isconnected(): w.start()
    
    rt = w.wss(strCodes,"creditrating")
    srs = pd.Series(rt.Data[0], index=codes)
    return srs

def frameStrategy(obj, 
                  start='2015/12/31',
                  end=None,
                  defineMethod='default',
                  selMethod=None,
                  weightMethod='average',
                  roundMethod='daily'):
    
    obj.DB['Close'].fillna(method='pad',inplace=True)
    
    def getStartLoc(obj, date):

        if date in obj.DB['Amt'].index:
            
            i = obj.DB['Amt'].index.get_loc(date)
        else:
            fakeIndex = obj.DB['Amt'].index.map(str2dt)
            i = fakeIndex.get_loc(str2dt(date), method='ffill')
    
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
    

def weaknessPerformance(obj):
    
    dfRet = pd.DataFrame(index=obj.DB['Amt'].index, columns=['StrbPrem', 'Ten-Days Change'])
    
    codes = obj._excludeSpecial()
    
    srsCredit = getCredit(obj._excludeSpecial(hasEB=0))
    srsIssue = get_issueamount(obj._excludeSpecial(0))
    
    t = srsCredit.apply(lambda x: True if x in ['AAA','AA+'] else False) | (srsIssue >= 20.0)
    
    codes = list(t[t].index)
    
    for i,v in enumerate(obj.DB['Amt'].index):
        
        t = (obj.DB['Amt'].loc[v, codes] > 1000.0) & \
        (obj.DB['Outstanding'].loc[v, codes] > 2000000000) & \
        (obj.DB['ConvPrem'].loc[v, codes] > 35.0) 
        
        t = list(t[t].index)
        
        dfRet.loc[v, 'StrbPrem'] = obj.DB['StrbPrem'].loc[v, t].mean()
    try:
        dfRet['Ten-Days Change'] = pd.rolling_apply(dfRet['StrbPrem'],10,lambda arr: arr[-1] - arr[0])
    except:
        dfRet['Ten-Days Change'] = dfRet['StrbPrem'].rolling(10).apply(lambda arr: arr[-1] - arr[0])
    
    dfRet["10Days Avg"] = dfRet['Ten-Days Change'].rolling(10).mean()
    return dfRet
  

def plottingSummary(obj, fields, start, end=None):
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1,1,figsize=(17,10))
    
    if not end:
        
        end = obj.DB['Amt'].index[-1]

    if isinstance(fields, list):
        
        if not len(fields) == 2:
            
            raise ValueError('len of fields must be 2')
        else:
            
            ax.plot(obj.Indicator[fields[0]][start:end], color='teal', linewidth=2, label=fields[0])
            
            span = obj.Indicator[fields[0]][start:end].max() - obj.Indicator[fields[0]][start:end].min()
            
            upperLim = obj.Indicator[fields[0]][start:end].max() + span * 0.2
            downLim = obj.Indicator[fields[0]][start:end].min() - span * 0.2
            
            ax.set_ylim([downLim, upperLim])
            ax.legend(loc='upper left')
            
            ax2 = ax.twinx()
            
            ax2.plot(obj.Indicator[fields[1]][start:end], color='darkred', linewidth=2, label=fields[1])
            ax2.legend(loc='upper right')
            span = obj.Indicator[fields[1]][start:end].max() - obj.Indicator[fields[1]][start:end].min()
            upperLim = obj.Indicator[fields[1]][start:end].max() + span * 0.2
            downLim = obj.Indicator[fields[1]][start:end].min() - span * 0.2
            
            ax2.set_ylim([downLim, upperLim])
            
            ro = int(round(len(obj.Indicator[fields[0]][start:end]) / 6.0))
            ax.set_xticks(list(range(0, len(obj.Indicator[fields[0]][start:end]), ro)))
            ax.set_xticklabels(obj.Indicator[fields[0]][start:end].index[::ro])                        
              
    elif isinstance(fields, str):
        
        ax.plot(obj.Indicator[fields][start:end], color='teal', linewidth=2, label=fields)

        span = obj.Indicator[fields][start:end].max() - obj.Indicator[fields][start:end].min()
        
        upperLim = obj.Indicator[fields][start:end].max() + span * 0.2
        downLim = obj.Indicator[fields][start:end].min() - span * 0.2
        
        ax.set_ylim([downLim, upperLim])
        ax.legend(loc='upper left')
        
        ro = int(round(len(obj.Indicator[fields][start:end]) / 6.0))
        ax.set_xticks(list(range(0, len(obj.Indicator[fields][start:end]), ro)))
        ax.set_xticklabels(obj.Indicator[fields][start:end].index[::ro])          
        
    fig.show()
    
    return fig
  

def cbName(lstCodes,outPut='s'):
    strCodes = ','.join(lstCodes)
    _windChecker()
    
    o = w.wss(strCodes, 'sec_name')
    
    if outPut == 's':
        return ','.join(o.Data[0])
    else:
        return o.Data[0]
    
def objSummary(obj):
    # 转债的汇总函数
    
    mat01 = obj.DB["Amt"].applymap(lambda x : 1.0 if x > 0 else 0.0)
    # 区分妖债与否
    _yaozhai = obj.DB["Close"].applymap(lambda x: 1.0 if x > 130 else 0) * \
    obj.DB["ConvPrem"].applymap(lambda x: 1.0 if x > 30.0 else 0.0)
    mat01NoYaozhai = mat01 * (1.0 - _yaozhai)
    
    lstMean = ["Close", "ConvPrem", "YTM", "StrbPrem", "ImpliedVol"]
    lstSum = ["Amt", "Outstanding"]
    
    dfSummary = pd.DataFrame(index=obj.DB["Amt"].index)
    dfSummary["Count"] = mat01.sum(axis=1)
    dfSummaryNoYaozhai = pd.DataFrame(index=obj.DB["Amt"].index)
    dfSummaryNoYaozhai["Count"] = mat01NoYaozhai.sum(axis=1)
    
    for f in lstMean:
        dfSummary[f] = (obj.DB[f] * mat01).sum(axis=1) / mat01.sum(axis=1)
        dfSummaryNoYaozhai[f] = (obj.DB[f] * mat01NoYaozhai).sum(axis=1) / mat01NoYaozhai.sum(axis=1)
    
    for f in lstSum:
        dfSummary[f] = (obj.DB[f] * mat01).sum(axis=1)
        dfSummaryNoYaozhai[f] = (obj.DB[f] * mat01NoYaozhai).sum(axis=1)
    
    return dfSummary, dfSummaryNoYaozhai
