import datetime as dt
import pandas as pd
import numpy as np
from WindPy import w

def getCodeList():
    if not w.isconnected(): w.start()
    _ , dfIssue = w.wset("cbissue", "startdate=2015-01-01;enddate=2022-12-31", usedf=True)
    return list(dfIssue.loc[dfIssue["issue_type"] != "私募"].index)


def readTable(codes, field, start, end, others=""):
    _, df = w.wsd(",".join(codes), field, start, end, others, usedf=True)
    
    if pd.to_datetime(start) == pd.to_datetime(end):
        
        dfRet = pd.DataFrame(columns=df.index)
        dfRet.loc[start] = df.iloc[:, 0]
        return dfRet
    
    else:
        return df
    


def readFromSQL(codes, field, start, end, others=""):
    # 根据自己的数据库写吧，我这里直接放这个raise了
    raise NotImplementedError


def tblUpdate(df, end, field, method="wind-api"):
    # end为截止日期
    # method是为其他数据接口如同花顺、SQL库等，这里不展开
    codes = df.columns
    
    if method == "wind-api":
        
        dates = w.tdays(df.index[-1], end).Data[0]
        
        if len(dates) > 1:
            
            kwargs = "rfIndex=1" if field == "impliedvol" else None
            dfNew = readTable(codes, field, dates[1], dates[-1], kwargs)
            dfNew.index = pd.to_datetime(dfNew.index)
            
            df = df.append(dfNew)
            
            return df
        else:
            
            print("不用更新")
            return df        
        
    elif method == "sql":
        
        dfNew = readFromSQL(codes, field, df.index[-1], end).iloc[1:]
        dfNew.index = pd.to_datetime(dfNew.index)
        if len(dfNew) > 0:
            df = df.append(dfNew)
            return df
        
        else:
            print("不用更新")
            return df
        


class cb_data(object):

    def __init__(self):
        self.DB = {} # DB为字典，准备装载各维度的数据
        self.loadData() # loadData后续定义
        
        
    def loadData(self):

        self.dfParams = pd.read_excel("参数.xlsx", index_col=0)
        for k, v in self.dfParams.iterrows():
            df = pd.read_csv(v["文件名"], index_col=0)
            df.index = pd.to_datetime(df.index)
            self.DB[k] = df
        
        self.panel = pd.read_excel("静态数据.xlsx", index_col=0)
    
    
    def __getitem__(self, key):
        return self.DB[key] if key in self.DB.keys() else None
    
    
    def __getattr__(self, key):
        return self.DB[key] if key in self.DB.keys() else None
    
    
    @property
    def date(self):
        return self.DB["Amt"].index[-1]
    
    
    @property
    def codes(self):
        return list(self.DB["Amt"].columns)
    
    
    @property
    def codes_active(self):
        srs = self.DB["Amt"].loc[self.date, self.codes]
        return list(srs[srs > 0].index)    
    
    
    def update(self, end, method="wind-api"):
        for k, v in self.dfParams.iterrows():
            df = self.DB[k]
            df = tblUpdate(df, end, v["字段(Wind)"], method)
            self.DB[k] = df
            print(f'{k} 更新已完成')
            
            
    def insertNewKey(self, new_codes, method='wind-api'):
        
        for key,value in self.DB.items():
           
            diff = list(set(new_codes) - set(self.DB.keys()))
                                                                                                
            if diff:                                        
                    field = self.dfParams.loc[key, '字段(Wind)']
        
                    start = self.DB[key].index[0] ; end = self.DB[key].index[-1]
                    
                    if method == "wind-api":
                        kwargs = "rfIndex=1" if field == "impliedvol" else None
                        df = readTable(diff, field, start, end, kwargs)
                        
                    elif method == "sql":
                        df = readFromSQL(diff, field, start, end)
                        
                    value = value.join(df)
    
                    self.DB[key] = value
                        
            self.updatePanelData(new_codes)
    
    
    def readPanel(self, codes=None):
        
        date = pd.to_datetime(self.date).strftime("%Y%m%d")
        if codes is None : codes = self.codes
        
        dfParams = pd.read_excel("静态参数.xlsx", index_col=0)
        
        _, df = w.wss(codes, ",".join(dfParams["字段(Wind)"]),
                      f"tradedate={date}", usedf=True)
        
        df.columns = list(dfParams.index)
        
        return df
    
    
    def updatePanelData(self, new_codes=None):
        
        if new_codes is None: new_codes = self.codes
        
        diff = list(set(new_codes) - set(self.panel.index))
        
        if diff:
            dfNew = self.readPanel(diff)
            self.panel = self.panel.append(dfNew)
    
    
    @property
    def matTrading(self):
        return self["Amt"].applymap(lambda x: 1 if x > 0 else np.nan)
    
    
    @property
    def matNormal(self):
        
        matTurn = self.DB["Amt"] * 10000.0 / self.DB["Outstanding"] / self.DB["Close"]
        
        matEx = (matTurn.applymap(lambda x: 1 if x > 100 else np.nan) * \
             self.DB["Close"].applymap(lambda x: 1 if x > 135 else np.nan) * \
             self.DB["ConvPrem"].applymap(lambda x: 1 if x >35 else np.nan)).applymap(lambda x: 1 if x != 1 else np.nan)
        
        return self.matTrading * matEx
    
    
    def output(self, prefix=""):
        for k , v in self.DB.items():
            v.to_csv(prefix = self.dfParams.loc[k, "文件名"])
        self.panel.to_excel("静态数据.xlsx")
