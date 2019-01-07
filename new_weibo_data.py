# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 22:48:03 2019

@author: 25493
"""

import pandas as pd
import numpy as np 
from datetime import timedelta
import re
from sklearn.ensemble import RandomForestRegressor as rfc

def read_csv():
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
    weibo_train_data = pd.read_csv("datalab/6336/weibo_train_data.csv" ,sep = "\t",parse_dates=['time'],date_parser = dateparse,names = ['uid','mid','time','forward_count','comment_count','like_count','content'],nrows = 10000)
    weibo_predict_data = pd.read_csv("datalab/6336/weibo_predict_data.csv",sep = "\t",parse_dates=['time'],date_parser = dateparse,names = ['uid','mid','time','content'])
    weibo_train_data = weibo_train_data.sort_values(by="time")
    weibo_predict_data = weibo_predict_data.sort_values(by="time")
    return weibo_train_data,weibo_predict_data
def big_extract(weibo_train_data,weibo_predict_data):
    #-----------提取hot_topic_time_delta hot_topic_first----------
    #合成big_data
    weibo_train_data_X = weibo_train_data[['uid','mid','time','content','hot_topic']]
    weibo_train_data_y = weibo_train_data[["forward_count","comment_count","like_count"]]
    tr_l = len(weibo_train_data_X)
    big_data = pd.concat([weibo_train_data_X,weibo_predict_data])
    
    #提取hot_topic属性
    big_data["hot_topic"] = big_data["content"].astype("str").str.split(r"#").map(lambda x:x[1] if len(x)>2 else np.nan)            
    #提取http_value属性
    pattern = re.compile(r"http://t.cn/\w+\s+")
    big_data["http_values"] = big_data["content"].astype("str").str.findall(pattern)
    big_data["http_values"] = big_data["http_values"].map(lambda x:x[0] if len(x)!=0 else -1)
    
    #离最小时间的时间间隔
    min_date = big_data["time"].min()
    big_data["seconds"] = big_data["time"].map(lambda x:(x-min_date).seconds)

    #big_data find "\\"
    big_data["is_discuss"] = big_data["content"].str.contains(r"\\")

    #big_data "\\" count
    big_data["is_discuss_count"] = big_data["content"].str.count(r"\\") - big_data["content"].str.count(r"http://")

    #hot_topic最小时间
    t = big_data.groupby("hot_topic")["time"].min().to_frame().reset_index()
    t = t.rename(columns = {"time":"min_time"})
    #hot_topic最小时间合并入big_data
    big_data = pd.merge(big_data,t,how = "left",on = "hot_topic")
    
    #离hot_topic最小时间的间隔
    big_data["hot_topic_time_delta"] = big_data["time"] - big_data["min_time"]
    big_data["hot_topic_time_delta"] = big_data["hot_topic_time_delta"].map(lambda x:x.seconds/60)
    
    #是否为最小时间
    big_data["hot_topic_first"] = big_data["min_time"] ==big_data["time"]
    
    #星期几
    big_data["weekday"] = big_data["time"].map(lambda x:x.weekday()) 
    
    #视频
    big_data["video"] = weibo_train_data.content.str.contains("视频").fillna(False)
    
    #一天中小时
    big_data["hour"] = big_data.time.map(lambda x:x.hour)
    
    #是否有标题
    big_data["have_title"] = big_data["content"].str.contains("【")
    
    big_data["week_of_year"] = big_data["time"].isocalendar()[1]
    this_week_hot = big_data.groupby("week_of_year")["mid"].count()
    big_data["this_week_hot"] = big_data["week_of_year"].map(this_week_hot)
    
    big_data["week_of_year-1"] = big_data["week_of_year"] - 1
    next_week_hot = big_data.groupby("week_of_year-1")["mid"].count()
    big_data["next_week_hot"] = big_data["week_of_year"].map(next_week_hot)
    
    big_data["week_of_year+1"] = big_data["week_of_year"] + 1
    last_week_hot = big_data.groupby("week_of_year+1")["mid"].count()
    big_data["last_week_hot"] = big_data["week_of_year"].map(last_week_hot)
    
    big_data = big_data.drop(["week_of_year-1","week_of_year+1"],axis = 1)
    
    #提取@的id
    big_data["at_name"] = big_data["content"].map(lambda x:str(x).split("@"))
    big_data["at_name"] = big_data["at_name"].map(lambda x:x[1] if len(x)>1 else np.nan)
    big_data["at_name4"] = big_data["at_name"].map(lambda x:x[:4] if isinstance(x,str) else np.nan)
    big_data["at_name5"] = big_data["at_name"].map(lambda x:x[:5] if isinstance(x,str) else np.nan)
    big_data["at_name6"] = big_data["at_name"].map(lambda x:x[:6] if isinstance(x,str) else np.nan)
    big_data["at_name7"] = big_data["at_name"].map(lambda x:x[:7] if isinstance(x,str) else np.nan)
    big_data["at_name8"] = big_data["at_name"].map(lambda x:x[:8] if isinstance(x,str) else np.nan)
    big_data["at_name9"] = big_data["at_name"].map(lambda x:x[:9] if isinstance(x,str) else np.nan)
    big_data["at_name10"] = big_data["at_name"].map(lambda x:x[:10] if isinstance(x,str) else np.nan)
    
    #content contains http
    big_data["content_con_http"] = big_data["content"].str.contains("http")
    #length of content
    big_data["content_len"] = big_data["content"].map(lambda x:len(x) if isinstance(x, str) else -1)
    #content contains count @
    big_data["content_count_at"] = big_data["content"].map(lambda x:x.count("@") if isinstance(x, str) else 0)
    #content contains @
    big_data["content_con_at"] = big_data["content"].str.contains("@")
    #content count count !
    big_data["content_count_exc"] = big_data["content"].map(lambda x:x.count("!") if isinstance(x, str) else 0)
    #content contains !
    big_data["content_con_exc"] = big_data["content"].str.contains("!")
    
    big_data["content_con_http"] = big_data["content_con_http"].astype(int)
    big_data["content_con_at"] = big_data["content_con_at"].astype(int)
    big_data["content_con_exc"] = big_data["content_con_exc"].astype(int)
    big_data["have_title"] = big_data["have_title"].astype(int)
    
    #合并
    weibo_train_data_X = big_data[:tr_l]
    weibo_predict_data = big_data[tr_l:]
    weibo_train_data = pd.concat([weibo_train_data_X,weibo_train_data_y],axis = 1)
    return weibo_train_data,weibo_predict_data
#x_train1 = extract_feature_data[extract_feature_data["time"]>=pd.datetime(2015,2,1) && extract_feature_data["time"]<=pd.datetime(2015,4,30)]
"""
split
train1 2.1-----4.30 > 5.1----5.31 test1
train2 3.1-----5.30 > 6.1----6.30 test2
train3 5.1-----7.31 > 8.1----8.31 test3（weibo_predict_data）
"""
def split_data(weibo_train_data,weibo_predict_data):
    train1 = weibo_train_data[(weibo_train_data["time"] >= pd.datetime(2015,2,1)) & (weibo_train_data["time"] <= pd.datetime(2015,2,28))]
    train2 = weibo_train_data[(weibo_train_data["time"] >= pd.datetime(2015,4,1)) & (weibo_train_data["time"] <= pd.datetime(2015,4,30))]
    train3 = weibo_train_data[(weibo_train_data["time"] >= pd.datetime(2015,7,1)) & (weibo_train_data["time"] <= pd.datetime(2015,7,31))]
                            
    test1 = weibo_train_data[(weibo_train_data["time"] >= pd.datetime(2015,3,1)) & (weibo_train_data["time"] <= pd.datetime(2015,3,31))]
    test2 = weibo_train_data[(weibo_train_data["time"] >= pd.datetime(2015,5,1)) & (weibo_train_data["time"] <= pd.datetime(2015,5,31))]
    test3 = weibo_predict_data
    return ((train1,test1),(train2,test2),(train3,test3))

def extraxt_merge_feature(train,test):
    #Processing "uid" of data by train
    #mean of forward_count group by uid
    uid_feature = train["uid"].to_frame()
    uid_mean_forward = train.groupby("uid")["forward_count"].mean().to_frame()
 
    #mean of comment_count group by uid
    uid_mean_comment = train.groupby("uid")["comment_count"].mean().to_frame()
 
    #mean of like_count group by uid
    uid_mean_like = train.groupby("uid")["like_count"].mean().to_frame()
 
    #all uid
    uid_count = train.groupby("uid")["mid"].count().to_frame()
 
    #merge
    uid_mean_forward.rename(columns = {"forward_count":"uid_mean_forward"},inplace = True)
    uid_mean_comment.rename(columns = {"comment_count":"uid_mean_comment"},inplace = True)
    uid_mean_like.rename(columns = {"like_count":"uid_mean_like"},inplace = True)
    uid_count.rename(columns = {"mid":"uid_count"},inplace = True)
    train = pd.merge(train,uid_mean_forward,left_on = "uid",right_index = True)
    train = pd.merge(train,uid_mean_comment,left_on = "uid",right_index = True)
    train = pd.merge(train,uid_mean_like,left_on = "uid",right_index = True)
    train = pd.merge(train,uid_count,left_on = "uid",right_index = True)
    
    #Processing "uid" of data by test1
    #mean of forward_count group by uid
    uid_feature = test["uid"].to_frame()
    #merge
    uid_feature = pd.merge(uid_feature,uid_mean_forward,left_on = "uid",right_index = True)
    uid_feature = pd.merge(uid_feature,uid_mean_comment,left_on = "uid",right_index = True)
    uid_feature = pd.merge(uid_feature,uid_mean_like,left_on = "uid",right_index = True)
    test1_uid_feature = pd.merge(uid_feature,uid_count,left_on = "uid",right_index = True)
    test1_uid_feature.columns = ["uid","uid_mean_forward","uid_mean_comment","uid_mean_like","uid_count"]
    
    #train hot_topic_mean
    hot_topic_forward_mean = train["forward_count"].groupby("hot_topic").mean().to_frame().reset_index()
    hot_topic_comment_mean = train["comment_count"].groupby("hot_topic").mean().to_frame().reset_index()
    hot_topic_like_mean = train["like_count"].groupby("hot_topic").mean().to_frame().reset_index()
    hot_topic_count = train["mid"].groupby("hot_topic").count().to_frame().reset_index()
    hot_topic_forward_mean.rename(columns = {"forward_count":"hot_topic_forward_mean"},inplace = True)
    hot_topic_comment_mean.rename(columns = {"comment_count":"hot_topic_comment_mean"},inplace = True)
    hot_topic_like_mean.rename(columns = {"like_count":"hot_topic_like_mean"},inplace = True)
    hot_topic_count.rename(columns = {"mid":"hot_topic_count"},inplace = True)
    train = pd.merge(train,hot_topic_forward_mean,how = "left",on = "hot_topic")
    train = pd.merge(train,hot_topic_comment_mean,how = "left",on = "hot_topic")
    train = pd.merge(train,hot_topic_like_mean,how = "left",on = "hot_topic")
    train = pd.merge(train,hot_topic_count,how = "left",on = "hot_topic")
    
    #test1 hot_topic_mean
    test = pd.merge(test,hot_topic_forward_mean,how = "left",on = "hot_topic")
    test = pd.merge(test,hot_topic_comment_mean,how = "left",on = "hot_topic")
    test = pd.merge(test,hot_topic_like_mean,how = "left",on = "hot_topic")
    test = pd.merge(test,hot_topic_count,how = "left",on = "hot_topic")
    
    #train uid_hot_topic_mean
    t1 = train[["uid","hot_topic","mid"]].groupby(["uid","hot_topic"]).count().reset_index()
    t2 = train[["uid","hot_topic","forward_count"]].groupby(["uid","hot_topic"]).mean().reset_index()
    t3 = train[["uid","hot_topic","comment_count"]].groupby(["uid","hot_topic"]).mean().reset_index()
    t4 = train[["uid","hot_topic","like_count"]].groupby(["uid","hot_topic"]).mean().reset_index()
    
    t1.rename(columns = {"mid":"uid_hot_topic_count"},inplace = True)
    t2.rename(columns = {"forward_count":"uid_hot_topic_forward_mean"},inplace = True)
    t3.rename(columns = {"comment_count":"uid_hot_topic_comment_mean"},inplace = True)
    t4.rename(columns = {"like_count":"uid_hot_topic_like_mean"},inplace = True)
    
    train = pd.merge(train,t1,how = "left",on = ["uid","hot_topic"])
    train = pd.merge(train,t2,how = "left",on = ["uid","hot_topic"])
    train = pd.merge(train,t3,how = "left",on = ["uid","hot_topic"])
    train = pd.merge(train,t4,how = "left",on = ["uid","hot_topic"])
    
    #test uid_hot_topic_mean
    test = pd.merge(test,t1,how = "left",on = ["uid","hot_topic"])
    test = pd.merge(test,t2,how = "left",on = ["uid","hot_topic"])
    test = pd.merge(test,t3,how = "left",on = ["uid","hot_topic"])
    test = pd.merge(test,t4,how = "left",on = ["uid","hot_topic"])
    
    #train uid_hour
    t1 = train[["uid","hour","mid"]].groupby(["uid","hour"]).count().reset_index()
    t2 = train[["uid","hour","forward_count"]].groupby(["uid","hour"]).mean().reset_index()
    t3 = train[["uid","hour","comment_count"]].groupby(["uid","hour"]).mean().reset_index()
    t4 = train[["uid","hour","like_count"]].groupby(["uid","hour"]).mean().reset_index()
    
    t1.rename(columns = {"mid":"uid_hour_count"},inplace = True)
    t2.rename(columns = {"forward_count":"uid_hour_forward_mean"},inplace = True)
    t3.rename(columns = {"comment_count":"uid_hour_comment_mean"},inplace = True)
    t4.rename(columns = {"like_count":"uid_hour_like_mean"},inplace = True)
    
    train = pd.merge(train,t1,how = "left",on = ["uid","hour"])
    train = pd.merge(train,t2,how = "left",on = ["uid","hour"])
    train = pd.merge(train,t3,how = "left",on = ["uid","hour"])
    train = pd.merge(train,t4,how = "left",on = ["uid","hour"])
    
    #test1 uid_hour
    test = pd.merge(test,t1,how = "left",on = ["uid","hour"])
    test = pd.merge(test,t2,how = "left",on = ["uid","hour"])
    test = pd.merge(test,t3,how = "left",on = ["uid","hour"])
    test = pd.merge(test,t4,how = "left",on = ["uid","hour"])
    
    t1 = train[["at_name4","like_count"]].groupby("at_name4").mean()["like_count"]
    
    t1 = train[["at_name4","like_count"]].groupby("at_name4").mean()["like_count"]
    t2 = train[["at_name4","forward_count"]].groupby("at_name4").mean()["forward_count"]
    t3 = train[["at_name4","comment_count"]].groupby("at_name4").mean()["comment_count"]
    t4 = train[["at_name4","like_count"]].groupby("at_name4").count()["like_count"]
    train["at_name4_like_count"] = train["at_name4"].map(t1)
    train["at_name4_forward_count"] = train["at_name4"].map(t2)
    train["at_name4_comment_count"] = train["at_name4"].map(t3)
    train["at_name4_count"] = train["at_name4"].map(t4)
    
    test["at_name4_like_count"] = test["at_name4"].map(t1)
    test["at_name4_forward_count"] = test["at_name4"].map(t2)
    test["at_name4_comment_count"] = test["at_name4"].map(t3)
    test["at_name4_count"] = test["at_name4"].map(t4)
    
    t1 = train[["at_name5","like_count"]].groupby("at_name5").mean()["like_count"]
    t2 = train[["at_name5","forward_count"]].groupby("at_name5").mean()["forward_count"]
    t3 = train[["at_name5","comment_count"]].groupby("at_name5").mean()["comment_count"]
    t4 = train[["at_name5","like_count"]].groupby("at_name5").count()["like_count"]
    
    train["at_name5_like_count"] = train["at_name5"].map(t1)
    train["at_name5_forward_count"] = train["at_name5"].map(t2)
    train["at_name5_comment_count"] = train["at_name5"].map(t3)
    train["at_name5_count"] = train["at_name5"].map(t4)
    
    test["at_name5_like_count"] = test["at_name5"].map(t1)
    test["at_name5_forward_count"] = test["at_name5"].map(t2)
    test["at_name5_comment_count"] = test["at_name5"].map(t3)
    test["at_name5_count"] = test["at_name5"].map(t4)
    
    t1 = train[["at_name6","like_count"]].groupby("at_name6").mean()["like_count"]
    t2 = train[["at_name6","forward_count"]].groupby("at_name6").mean()["forward_count"]
    t3 = train[["at_name6","comment_count"]].groupby("at_name6").mean()["comment_count"]
    t4 = train[["at_name6","like_count"]].groupby("at_name6").count()["like_count"]
    
    train["at_name6_like_count"] = train["at_name6"].map(t1)
    train["at_name6_forward_count"] = train["at_name6"].map(t2)
    train["at_name6_comment_count"] = train["at_name6"].map(t3)
    train["at_name6_count"] = train["at_name6"].map(t4)
    
    test["at_name6_like_count"] = test["at_name6"].map(t1)
    test["at_name6_forward_count"] = test["at_name6"].map(t2)
    test["at_name6_comment_count"] = test["at_name6"].map(t3)
    test["at_name6_count"] = test["at_name6"].map(t4)
    
    t1 = train[["at_name7","like_count"]].groupby("at_name7").mean()["like_count"]
    t2 = train[["at_name7","forward_count"]].groupby("at_name7").mean()["forward_count"]
    t3 = train[["at_name7","comment_count"]].groupby("at_name7").mean()["comment_count"]
    t4 = train[["at_name7","like_count"]].groupby("at_name7").count()["like_count"]
    
    train["at_name7_like_count"] = train["at_name7"].map(t1)
    train["at_name7_forward_count"] = train["at_name7"].map(t2)
    train["at_name7_comment_count"] = train["at_name7"].map(t3)
    train["at_name7_count"] = train["at_name7"].map(t4)
    
    test["at_name7_like_count"] = test["at_name7"].map(t1)
    test["at_name7_forward_count"] = test["at_name7"].map(t2)
    test["at_name7_comment_count"] = test["at_name7"].map(t3)
    test["at_name7_count"] = test["at_name7"].map(t4)
    
    t1 = train[["at_name8","like_count"]].groupby("at_name8").mean()["like_count"]
    t2 = train[["at_name8","forward_count"]].groupby("at_name8").mean()["forward_count"]
    t3 = train[["at_name8","comment_count"]].groupby("at_name8").mean()["comment_count"]
    t4 = train[["at_name8","like_count"]].groupby("at_name8").count()["like_count"]
    
    train["at_name8_like_count"] = train["at_name8"].map(t1)
    train["at_name8_forward_count"] = train["at_name8"].map(t2)
    train["at_name8_comment_count"] = train["at_name8"].map(t3)
    train["at_name8_count"] = train["at_name8"].map(t4)
    
    test["at_name8_like_count"] = test["at_name8"].map(t1)
    test["at_name8_forward_count"] = test["at_name8"].map(t2)
    test["at_name8_comment_count"] = test["at_name8"].map(t3)
    test["at_name8_count"] = test["at_name8"].map(t4)
    
    t1 = train[["at_name9","like_count"]].groupby("at_name9").mean()["like_count"]
    t2 = train[["at_name9","forward_count"]].groupby("at_name9").mean()["forward_count"]
    t3 = train[["at_name9","comment_count"]].groupby("at_name9").mean()["comment_count"]
    t4 = train[["at_name9","like_count"]].groupby("at_name9").count()["like_count"]
    
    train["at_name9_like_count"] = train["at_name9"].map(t1)
    train["at_name9_forward_count"] = train["at_name9"].map(t2)
    train["at_name9_comment_count"] = train["at_name9"].map(t3)
    train["at_name9_count"] = train["at_name9"].map(t4)
    
    test["at_name9_like_count"] = test["at_name9"].map(t1)
    test["at_name9_forward_count"] = test["at_name9"].map(t2)
    test["at_name9_comment_count"] = test["at_name9"].map(t3)
    test["at_name9_count"] = test["at_name9"].map(t4)
    
    t1 = train[["at_name10","like_count"]].groupby("at_name10").mean()["like_count"]
    t2 = train[["at_name10","forward_count"]].groupby("at_name10").mean()["forward_count"]
    t3 = train[["at_name10","comment_count"]].groupby("at_name10").mean()["comment_count"]
    t4 = train[["at_name10","like_count"]].groupby("at_name10").count()["like_count"]
    
    train["at_name10_like_count"] = train["at_name10"].map(t1)
    train["at_name10_forward_count"] = train["at_name10"].map(t2)
    train["at_name10_comment_count"] = train["at_name10"].map(t3)
    train["at_name10_count"] = train["at_name10"].map(t4)
    
    test["at_name10_like_count"] = test["at_name10"].map(t1)
    test["at_name10_forward_count"] = test["at_name10"].map(t2)
    test["at_name10_comment_count"] = test["at_name10"].map(t3)
    test["at_name10_count"] = test["at_name10"].map(t4)
    
    t1 = train[["http_values","mid"]].groupby(["http_values"]).count().reset_index()
    t2 = train[["http_values","forward_count"]].groupby(["http_values"]).mean().reset_index()
    t3 = train[["http_values","comment_count"]].groupby(["http_values"]).mean().reset_index()
    t4 = train[["http_values","like_count"]].groupby(["http_values"]).mean().reset_index()
    
    t1.rename(columns = {"mid":"http_values_count"},inplace = True)
    t2.rename(columns = {"forward_count":"http_values_forward_mean"},inplace = True)
    t3.rename(columns = {"comment_count":"http_values_comment_mean"},inplace = True)
    t4.rename(columns = {"like_count":"http_values_like_mean"},inplace = True)
    
    train = pd.merge(train,t1,how = "left",on = ["http_values"])
    train = pd.merge(train,t2,how = "left",on = ["http_values"])
    train = pd.merge(train,t3,how = "left",on = ["http_values"])
    train = pd.merge(train,t4,how = "left",on = ["http_values"])
    
    test = pd.merge(test,t1,how = "left",on = ["http_values"])
    test = pd.merge(test,t2,how = "left",on = ["http_values"])
    test = pd.merge(test,t3,how = "left",on = ["http_values"])
    test = pd.merge(test,t4,how = "left",on = ["http_values"])
    return train,test

def input_data(data):
    result = []
    for train,test in data:
        extract_train,extract_test = extraxt_merge_feature(train,test)
        result.append((extract_train,extract_test))
    return result

def train_split(data,flag = False):
    
    if flag:
        #extract_feature =["uid_mean_forward","uid_mean_comment","uid_mean_like","uid_count","content_con_http","content_len","content_count_at","content_con_at","content_count_exc","content_con_exc","hot_topic_forward_mean","hot_topic_comment_mean","hot_topic_like_mean","hot_topic_count","hot_topic_time_delta","hot_topic_first"] 
        extract_feature = ["uid_mean_forward","uid_mean_comment","uid_mean_like","uid_count","content_con_http","content_count_at","content_con_at","content_con_exc","hot_topic_forward_mean","hot_topic_comment_mean","hot_topic_like_mean","hot_topic_count","hot_topic_time_delta","uid_hot_topic_count","uid_hot_topic_forward_mean","uid_hot_topic_comment_mean","uid_hot_topic_like_mean","weekday","is_discuss_count","video","hour","uid_hour_count","uid_hour_forward_mean","uid_hour_like_mean","uid_hour_comment_mean","hot_topic_first","have_title","week_of_year","this_week_hot","next_week_hot","last_week_hot","at_name4_like_count","at_name4_forward_count","at_name4_comment_count","at_name4_count","at_name5_like_count","at_name5_forward_count","at_name5_comment_count","at_name5_count","at_name6_like_count","at_name6_forward_count","at_name6_comment_count","at_name6_count","at_name7_like_count","at_name7_forward_count","at_name7_comment_count","at_name7_count","at_name8_like_count","at_name8_forward_count","at_name8_comment_count","at_name8_count","at_name9_like_count","at_name9_forward_count","at_name9_comment_count","at_name9_count","at_name10_like_count","at_name10_forward_count","at_name10_comment_count","at_name10_count","seconds"]
        train,test = data.pop()
        x_train = train[extract_feature]
        y_train = train["forward_count","comment_count","comment_count"] 
        x_test = test[extract_feature]
        return x_train,y_train,x_test
    result = []
    data.pop()
    for train,test in data:
        x_train = train[extract_feature]
        y_train_forward_count = train["forward_count"] 
        y_train_comment_count = train["comment_count"]
        y_train_like_count = train["like_count"]
        
        x_test = test[extract_feature]
        y_test_forward_count = test["forward_count"]
        y_test_comment_count = test["comment_count"]
        y_test_like_count = test["like_count"]
        
        result.append(((x_train,y_train_forward_count,y_train_comment_count,y_train_like_count),\
                      (x_test,y_test_forward_count,y_test_comment_count,y_test_like_count)))
        return result
    
def train_metrics(x_train,y_train,x_test,y_test,flag = True):
    x_train = x_train.fillna(-1)
    x_test = x_test.fillna(-1)
    
    y_train_forward_count,y_train_comment_count,y_train_like_count = y_train
    y_test_forward_count,y_test_comment_count,y_test_like_count = y_test

    model =rfc()
    model.fit(x_train,y_train_forward_count)
    predict_forward_count = model.predict(x_test)
 
    model.fit(x_train,y_train_comment_count)
    predict_comment_count = model.predict(x_test)
 
    model.fit(x_train,y_train_like_count)
    predict_like_count = model.predict(x_test)

    f = np.abs(y_test_forward_count - predict_forward_count)/(y_test_forward_count+5)
    c = np.abs(y_test_comment_count - predict_comment_count)/(y_test_comment_count+3)
    l = np.abs(y_test_like_count - predict_like_count)/(y_test_like_count+3)
    pre = 1-0.5*f-0.25*c-0.25*l
    ci = y_test_forward_count+y_test_comment_count+y_test_like_count
    ci[ci>100] = 100
    
    pre = (pre-0.8).map(lambda x:1 if x>0 else 0)
    error = ((ci*pre).sum()+pre.sum())/(pre.sum()+len(pre))
    print(error)

    c = x_train.columns
    s = np.argsort(rfc.feature_importances_)
    feature_score = pd.DataFrame(columns=["feature","score"])
    feature_score["feature"] = c[s]
    feature_score["score"] = rfc.feature_importances_[s]
    feature_score
    
    return model

def predict(x_train,y_train,x_test):
    x_train = x_train.fillna(-1)
    x_test = x_test.fillna(-1)
    
    y_train_forward_count,y_train_comment_count,y_train_like_count = y_train

    model =rfc()
    model.fit(x_train,y_train_forward_count)
    predict_forward_count = model.predict(x_test)
 
    model.fit(x_train,y_train_comment_count)
    predict_comment_count = model.predict(x_test)
 
    model.fit(x_train,y_train_like_count)
    predict_like_count = model.predict(x_test)
    
    c = x_train.columns
    s = np.argsort(rfc.feature_importances_)
    feature_score = pd.DataFrame(columns=["feature","score"])
    feature_score["feature"] = c[s]
    feature_score["score"] = rfc.feature_importances_[s]
    feature_score
    return predict_forward_count,predict_comment_count,predict_like_count

def submit(predict_forward_count,predict_comment_count,predict_like_count):
    result = pd.DataFrame(columns=["uid","mid","forward_count","comment_count","like_count"])
    test = pd.read_csv("datalab/6336/weibo_predict_data.csv",sep = "\t",names = ['uid','mid','time','content'])
    result["uid"] = test["uid"]
    result["mid"] = test["mid"]
    result["forward_count"] = predict_forward_count
    result["comment_count"] = predict_comment_count
    result["like_count"] = predict_like_count


def start():
    weibo_train_data,weibo_predict_data = read_csv()
    weibo_train_data,weibo_predict_data = big_extract(weibo_train_data,weibo_predict_data)
    data = split_data(weibo_train_data,weibo_predict_data)
    extract_data_result = input_data(data)
    
    result = train_split(extract_data_result)
    
    x_train,y_train,x_test,y_test = result.pop()
    train_metrics(x_train,y_train,x_test,y_test,flag = True)
    
    x_train,y_train,x_test,y_test = result.pop()
    train_metrics(x_train,y_train,x_test,y_test,flag = True)
    
    x_train,y_train,x_test = train_split(extract_data_result,flag = True)
    predict_forward_count,predict_comment_count,predict_like_count = predict(x_train,y_train,x_test)
    
    submit(predict_forward_count,predict_comment_count,predict_like_count)

start()