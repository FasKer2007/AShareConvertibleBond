# AShareConvertibleBond
转债研究的日常,目前可以公开的最大限度,报告里基本有,这里仅供技术交流和复制粘贴 —— 杨冰
1. libCB3.py是基础的数据装载库，最重要的包括数据库变量cb_data，以及测算框架frameStrategy。使用时推荐import libCB3 as cb
**最新会有一个报告讲这些是干啥用的，可以参考**

2. strategy3.py有大量基础策略及相关应用，搭配frameStrategy
## libCB3
### cb.cb_data
一般令 obj = cb.cb_data()，有各类用法。
1. 调用数据库，现在支持：obj.DB["xxx"] 、 obj["xxx"] 、 obj.xxx来引用对应字段的DataFrame了。字段可以参考参数表，例如obj.Close就是收盘价的面板数据；
2. date、codes、codes_active： 最后交易日、转债代码以及当前交易的转债代码
3. obj.update、obj.insertNewkey: 更新数据、插入新的转债
4. obj.panel为静态数据，一个面板，有评级、条款什么的，可以自己改，看“静态参数.xlsx”


Examples：

obj2 = cb_data()

求均价，非异常样本

(obj2.matNormal * obj2.Close).apply(np.mean, axis=1)

求平价90~110元转债平均溢价率

(obj2.matNormal * obj2.ConvV.applymap(lambda x: 1 if 90 <= x < 110 else np.nan) *\
obj2.ConvPrem).apply(np.mean, axis=1)

求10日平均隐含波动率

(obj2.matNormal * obj2.ImpliedVol).apply(np.mean, axis=1).rolling(10).mean()

## strategy3
未完待续，不续别催
