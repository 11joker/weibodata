# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 22:48:03 2019

@author: 25493
"""

import pandas as pd
import numpy as np 
from datetime import timedelta
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
weibo_train_data = pd.read_csv("datalab/6336/weibo_train_data.csv" ,sep = "\t",parse_dates=['time'],date_parser = dateparse,names = ['uid','mid','time','forward_count','comment_count','like_count','content'])
weibo_predict_data = pd.read_csv("datalab/6336/weibo_predict_data.csv",sep = "\t",parse_dates=['time'],date_parser = dateparse,names = ['uid','mid','time','content'])
weibo_train_data = weibo_train_data.sort_values(by="time")
weibo_predict_data = weibo_predict_data.sort_values(by="time")

#提取hot_topic属性
hot_topic = weibo_train_data["content"].astype("str").str.split(r"#").map(lambda x:x[1] if len(x)>2 else np.nan)
hot_topic.name = "hot_topic"
weibo_train_data = pd.concat([weibo_train_data,hot_topic],axis=1)

weibo_predict_data["hot_topic"] = weibo_predict_data["content"].astype("str").str.split(r"#").map(lambda x:x[1] if len(x)>2 else np.nan)
                  
#提取http_value属性
import re
pattern = re.compile(r"http://t.cn/\w+\s+")
weibo_train_data["http_values"] = weibo_train_data["content"].astype("str").str.findall(pattern)
weibo_train_data["http_values"] = weibo_train_data["http_values"].map(lambda x:x[0] if len(x)!=0 else -1)

weibo_predict_data["http_values"] = weibo_predict_data["content"].astype("str").str.findall(pattern)
weibo_predict_data["http_values"] = weibo_predict_data["http_values"].map(lambda x:x[0] if len(x)!=0 else -1)

#-----------提取hot_topic_time_delta hot_topic_first----------
#合成big_data
weibo_train_data_X = weibo_train_data[['uid','mid','time','content','hot_topic']]
weibo_train_data_y = weibo_train_data[["forward_count","comment_count","like_count"]]
tr_l = len(weibo_train_data_X)
big_data = pd.concat([weibo_train_data_X,weibo_predict_data])

min_date = big_data["time"].min()
big_data["seconds"] = big_data["time"].map(lambda x:(x-min_date).seconds)

big_data["days"] = big_data["seconds"].map(lambda x:x//86400)

big_data["half_days"] = big_data["seconds"].map(lambda x:x//43200)
#big_data find "\\"
big_data["is_discuss"] = big_data["content"].str.contains(r"\\")

#big_data "\\" count
big_data["is_discuss_count"] = big_data["content"].str.count(r"\\") - big_data["content"].str.count(r"http://")

#hot_topic最小时间
t = big_data.groupby("hot_topic")["time"].min().to_frame().reset_index()
t = t.rename(columns = {"time":"min_time"})
#合并入big_data
big_data = pd.merge(big_data,t,how = "left",on = "hot_topic")

big_data["hot_topic_time_delta"] = big_data["time"] - big_data["min_time"]
big_data["hot_topic_time_delta"] = big_data["hot_topic_time_delta"].map(lambda x:x.seconds/60)

big_data["hot_topic_first"] = big_data["min_time"] ==big_data["time"]

big_data["weekday"] = big_data["time"].map(lambda x:x.weekday()) 

big_data["video"] = weibo_train_data.content.str.contains("视频").fillna(False)

big_data["hour"] = big_data.time.map(lambda x:x.hour)

big_data["have_title"] = big_data["content"].str.contains("【")

big_data["week_of_year"] = big_data["time"].dt.dayofweek
this_week_hot = big_data.groupby("week_of_year")["mid"].count()
big_data["this_week_hot"] = big_data["week_of_year"].map(this_week_hot)

big_data["week_of_year-1"] = big_data["week_of_year"] - 1
next_week_hot = big_data.groupby("week_of_year-1")["mid"].count()
big_data["next_week_hot"] = big_data["week_of_year"].map(next_week_hot)

big_data["week_of_year+1"] = big_data["week_of_year"] + 1
last_week_hot = big_data.groupby("week_of_year+1")["mid"].count()
big_data["last_week_hot"] = big_data["week_of_year"].map(last_week_hot)

big_data = big_data.drop(["week_of_year-1","week_of_year+1"],axis = 1)


this_day_hot = big_data.groupby("days")["mid"].count()
big_data["this_day_hot"] = big_data["days"].map(this_day_hot)

big_data["days-1"] = big_data["days"] - 1
next_day_hot = big_data.groupby("days-1")["mid"].count()
big_data["next_day_hot"] = big_data["days"].map(next_day_hot)

big_data["days+1"] = big_data["days"] + 1
last_day_hot = big_data.groupby("days+1")["mid"].count()
big_data["last_week_hot"] = big_data["days"].map(last_day_hot)

big_data = big_data.drop(["days-1","days+1"],axis = 1)

big_data["at_name"] = big_data["content"].map(lambda x:str(x).split("@"))
big_data["at_name"] = big_data["at_name"].map(lambda x:x[1] if len(x)>1 else np.nan)
big_data["at_name2"] = big_data["at_name"].map(lambda x:x[:2] if isinstance(x,str) else np.nan)
big_data["at_name3"] = big_data["at_name"].map(lambda x:x[:3] if isinstance(x,str) else np.nan)
big_data["at_name4"] = big_data["at_name"].map(lambda x:x[:4] if isinstance(x,str) else np.nan)
big_data["at_name5"] = big_data["at_name"].map(lambda x:x[:5] if isinstance(x,str) else np.nan)
big_data["at_name6"] = big_data["at_name"].map(lambda x:x[:6] if isinstance(x,str) else np.nan)
big_data["at_name7"] = big_data["at_name"].map(lambda x:x[:7] if isinstance(x,str) else np.nan)
big_data["at_name8"] = big_data["at_name"].map(lambda x:x[:8] if isinstance(x,str) else np.nan)
big_data["at_name9"] = big_data["at_name"].map(lambda x:x[:9] if isinstance(x,str) else np.nan)
big_data["at_name10"] = big_data["at_name"].map(lambda x:x[:10] if isinstance(x,str) else np.nan)

t = big_data[["at_name2","mid"]].groupby("at_name2").count()["mid"]
big_data["at_name2_count"] = big_data["at_name2"].map(t)

t = big_data[["at_name3","mid"]].groupby("at_name3").count()["mid"]
big_data["at_name3_count"] = big_data["at_name3"].map(t)

t = big_data[["at_name4","mid"]].groupby("at_name4").count()["mid"]
big_data["at_name4_count"] = big_data["at_name4"].map(t)

t = big_data[["at_name5","mid"]].groupby("at_name5").count()["mid"]
big_data["at_name5_count"] = big_data["at_name5"].map(t)

t = big_data[["at_name6","mid"]].groupby("at_name6").count()["mid"]
big_data["at_name6_count"] = big_data["at_name6"].map(t)

t = big_data[["at_name7","mid"]].groupby("at_name7").count()["mid"]
big_data["at_name7_count"] = big_data["at_name7"].map(t)

t = big_data[["at_name8","mid"]].groupby("at_name8").count()["mid"]
big_data["at_name8_count"] = big_data["at_name8"].map(t)

t = big_data[["at_name9","mid"]].groupby("at_name9").count()["mid"]
big_data["at_name9_count"] = big_data["at_name9"].map(t)

t = big_data[["at_name10","mid"]].groupby("at_name10").count()["mid"]
big_data["at_name10_count"] = big_data["at_name10"].map(t)

t = ["at_name2","at_name2_count","at_name3","at_name3_count","at_name4",
     "at_name4_count","at_name5","at_name5_count","at_name6","at_name6_count","at_name7","at_name7_count","at_name8",
     "at_name8_count","at_name9","at_name9_count","at_name10","at_name10_count"]

#提取@
def extract_feature(series):
    i = 0
    if (str(series["at_name3"])) == "nan" | (len(str(series["at_name3"]))) == 0:
            return np.nan
    while i < len(t):
        feature_name = t[i]
        feature_name = str(series[feature_name])
        i+=1
        if i>=len(t):
            return feature_name
        feature_count =t[i]
        feature_count = series[feature_count]
        w = str(feature_name[-1])
        i+=1
        if not ('\u4e00' <= w <= '\u9fff'):
            if not (("A"<= w <="Z") | ("a"<= w <="z")):
                if not ("0" <= w <= "9"):
                    if not ((w=="_") | (w=="-")):
                        return feature_name[:-1]
        if (w=="-") | (w=="_"):
            continue
            print(feature_count)
        if (feature_count<10) & (i==1):
            return np.nan
        if feature_count<10:
            return feature_name[:-1]
        
        
#big_data[:100].apply(extract_feature,axis = 1)        
big_data["at_name"] = big_data.apply(extract_feature,axis = 1)

weibo_train_data_X = big_data[:tr_l]
weibo_predict_data = big_data[tr_l:]

weibo_train_data = pd.concat([weibo_train_data_X,weibo_train_data_y],axis = 1)

t1 = weibo_train_data[["at_name","like_count"]].groupby("at_name").mean()["like_count"]
t2 = weibo_train_data[["at_name","forward_count"]].groupby("at_name").mean()["forward_count"]
t3 = weibo_train_data[["at_name","comment_count"]].groupby("at_name").mean()["comment_count"]
t4 = weibo_train_data[["at_name","like_count"]].groupby("at_name").count()["like_count"]
weibo_train_data["big_at_name_like_count"] = weibo_train_data["at_name"].map(t1)
weibo_train_data["big_at_name_forward_count"] = weibo_train_data["at_name"].map(t2)
weibo_train_data["big_at_name_comment_count"] = weibo_train_data["at_name"].map(t3)
weibo_train_data["big_at_name_count"] = weibo_train_data["at_name"].map(t4)

weibo_predict_data["big_at_name_like_count"] = weibo_predict_data["at_name"].map(t1)
weibo_predict_data["big_at_name_forward_count"] = weibo_predict_data["at_name"].map(t2)
weibo_predict_data["big_at_name_comment_count"] = weibo_predict_data["at_name"].map(t3)
weibo_predict_data["big_at_name_count"] = weibo_predict_data["at_name"].map(t4)

#x_train1 = extract_feature_data[extract_feature_data["time"]>=pd.datetime(2015,2,1) && extract_feature_data["time"]<=pd.datetime(2015,4,30)]
"""
split
train1 2.1-----4.30 > 5.1----5.31 test1
train2 3.1-----5.30 > 6.1----6.30 test2
train3 5.1-----7.31 > 8.1----8.31 test3（weibo_predict_data）
"""
train1 = weibo_train_data[(weibo_train_data["time"] >= pd.datetime(2015,2,1)) & (weibo_train_data["time"] <= pd.datetime(2015,2,28))]
train2 = weibo_train_data[(weibo_train_data["time"] >= pd.datetime(2015,4,1)) & (weibo_train_data["time"] <= pd.datetime(2015,4,30))]
train3 = weibo_train_data[(weibo_train_data["time"] >= pd.datetime(2015,7,1)) & (weibo_train_data["time"] <= pd.datetime(2015,7,31))]
                        
test1 = weibo_train_data[(weibo_train_data["time"] >= pd.datetime(2015,3,1)) & (weibo_train_data["time"] <= pd.datetime(2015,3,31))]
test2 = weibo_train_data[(weibo_train_data["time"] >= pd.datetime(2015,5,1)) & (weibo_train_data["time"] <= pd.datetime(2015,5,31))]
test3 = weibo_predict_data
del weibo_predict_data
del weibo_train_data  

#Processing "uid" of data by train1
#mean of forward_count group by uid
uid_feature = train1["uid"].to_frame()
uid_mean_forward = train1.groupby("uid")["forward_count"].mean().to_frame()
 
#mean of comment_count group by uid
uid_mean_comment = train1.groupby("uid")["comment_count"].mean().to_frame()
 
#mean of like_count group by uid
uid_mean_like = train1.groupby("uid")["like_count"].mean().to_frame()
 
#all uid
uid_count = train1.groupby("uid")["mid"].count().to_frame()
 
#merge
uid_feature = pd.merge(uid_feature,uid_mean_forward,left_on = "uid",right_index = True)
uid_feature = pd.merge(uid_feature,uid_mean_comment,left_on = "uid",right_index = True)
uid_feature = pd.merge(uid_feature,uid_mean_like,left_on = "uid",right_index = True)
train1_uid_feature = pd.merge(uid_feature,uid_count,left_on = "uid",right_index = True)
train1_uid_feature.columns = ["uid","uid_mean_forward","uid_mean_comment","uid_mean_like","uid_count"]

#Processing "uid" of data by test1
#mean of forward_count group by uid
uid_feature = test1["uid"].to_frame()
#merge
uid_feature = pd.merge(uid_feature,uid_mean_forward,left_on = "uid",right_index = True)
uid_feature = pd.merge(uid_feature,uid_mean_comment,left_on = "uid",right_index = True)
uid_feature = pd.merge(uid_feature,uid_mean_like,left_on = "uid",right_index = True)
test1_uid_feature = pd.merge(uid_feature,uid_count,left_on = "uid",right_index = True)
test1_uid_feature.columns = ["uid","uid_mean_forward","uid_mean_comment","uid_mean_like","uid_count"]

#train1 hot_topic_mean
hot_topic_forward_mean = train1["forward_count"].groupby(hot_topic).mean().to_frame().reset_index()
hot_topic_comment_mean = train1["comment_count"].groupby(hot_topic).mean().to_frame().reset_index()
hot_topic_like_mean = train1["like_count"].groupby(hot_topic).mean().to_frame().reset_index()
hot_topic_count = train1["mid"].groupby(hot_topic).count().to_frame().reset_index()
hot_topic_forward_mean.rename(columns = {"forward_count":"hot_topic_forward_mean"},inplace = True)
hot_topic_comment_mean.rename(columns = {"comment_count":"hot_topic_comment_mean"},inplace = True)
hot_topic_like_mean.rename(columns = {"like_count":"hot_topic_like_mean"},inplace = True)
hot_topic_count.rename(columns = {"mid":"hot_topic_count"},inplace = True)
train1 = pd.merge(train1,hot_topic_forward_mean,how = "left",on = "hot_topic")
train1 = pd.merge(train1,hot_topic_comment_mean,how = "left",on = "hot_topic")
train1 = pd.merge(train1,hot_topic_like_mean,how = "left",on = "hot_topic")
train1 = pd.merge(train1,hot_topic_count,how = "left",on = "hot_topic")

#test1 hot_topic_mean
test1 = pd.merge(test1,hot_topic_forward_mean,how = "left",on = "hot_topic")
test1 = pd.merge(test1,hot_topic_comment_mean,how = "left",on = "hot_topic")
test1 = pd.merge(test1,hot_topic_like_mean,how = "left",on = "hot_topic")
test1 = pd.merge(test1,hot_topic_count,how = "left",on = "hot_topic")

#train1 uid_hot_topic_mean
t1 = train1[["uid","hot_topic","mid"]].groupby(["uid","hot_topic"]).count().reset_index()
t2 = train1[["uid","hot_topic","forward_count"]].groupby(["uid","hot_topic"]).mean().reset_index()
t3 = train1[["uid","hot_topic","comment_count"]].groupby(["uid","hot_topic"]).mean().reset_index()
t4 = train1[["uid","hot_topic","like_count"]].groupby(["uid","hot_topic"]).mean().reset_index()

t1.rename(columns = {"mid":"uid_hot_topic_count"},inplace = True)
t2.rename(columns = {"forward_count":"uid_hot_topic_forward_mean"},inplace = True)
t3.rename(columns = {"comment_count":"uid_hot_topic_comment_mean"},inplace = True)
t4.rename(columns = {"like_count":"uid_hot_topic_like_mean"},inplace = True)

train1 = pd.merge(train1,t1,how = "left",on = ["uid","hot_topic"])
train1 = pd.merge(train1,t2,how = "left",on = ["uid","hot_topic"])
train1 = pd.merge(train1,t3,how = "left",on = ["uid","hot_topic"])
train1 = pd.merge(train1,t4,how = "left",on = ["uid","hot_topic"])

#test1 uid_hot_topic_mean
test1 = pd.merge(test1,t1,how = "left",on = ["uid","hot_topic"])
test1 = pd.merge(test1,t2,how = "left",on = ["uid","hot_topic"])
test1 = pd.merge(test1,t3,how = "left",on = ["uid","hot_topic"])
test1 = pd.merge(test1,t4,how = "left",on = ["uid","hot_topic"])

#train1 uid_hour
t1 = train1[["uid","hour","mid"]].groupby(["uid","hour"]).count().reset_index()
t2 = train1[["uid","hour","forward_count"]].groupby(["uid","hour"]).mean().reset_index()
t3 = train1[["uid","hour","comment_count"]].groupby(["uid","hour"]).mean().reset_index()
t4 = train1[["uid","hour","like_count"]].groupby(["uid","hour"]).mean().reset_index()

t1.rename(columns = {"mid":"uid_hour_count"},inplace = True)
t2.rename(columns = {"forward_count":"uid_hour_forward_mean"},inplace = True)
t3.rename(columns = {"comment_count":"uid_hour_comment_mean"},inplace = True)
t4.rename(columns = {"like_count":"uid_hour_like_mean"},inplace = True)

train1 = pd.merge(train1,t1,how = "left",on = ["uid","hour"])
train1 = pd.merge(train1,t2,how = "left",on = ["uid","hour"])
train1 = pd.merge(train1,t3,how = "left",on = ["uid","hour"])
train1 = pd.merge(train1,t4,how = "left",on = ["uid","hour"])

#test1 uid_hour
test1 = pd.merge(test1,t1,how = "left",on = ["uid","hour"])
test1 = pd.merge(test1,t2,how = "left",on = ["uid","hour"])
test1 = pd.merge(test1,t3,how = "left",on = ["uid","hour"])
test1 = pd.merge(test1,t4,how = "left",on = ["uid","hour"])


t1 = train1[["at_name","like_count"]].groupby("at_name").mean()["like_count"]
t2 = train1[["at_name","forward_count"]].groupby("at_name").mean()["forward_count"]
t3 = train1[["at_name","comment_count"]].groupby("at_name").mean()["comment_count"]
t4 = train1[["at_name","like_count"]].groupby("at_name").count()["like_count"]
train1["at_name_like_count"] = train1["at_name"].map(t1)
train1["at_name_forward_count"] = train1["at_name"].map(t2)
train1["at_name_comment_count"] = train1["at_name"].map(t3)
train1["at_name_count"] = train1["at_name"].map(t4)

test1["at_name_like_count"] = test1["at_name"].map(t1)
test1["at_name_forward_count"] = test1["at_name"].map(t2)
test1["at_name_comment_count"] = test1["at_name"].map(t3)
test1["at_name_count"] = test1["at_name"].map(t4)

t1 = train1[["http_values","mid"]].groupby(["http_values"]).count().reset_index()
t2 = train1[["http_values","forward_count"]].groupby(["http_values"]).mean().reset_index()
t3 = train1[["http_values","comment_count"]].groupby(["http_values"]).mean().reset_index()
t4 = train1[["http_values","like_count"]].groupby(["http_values"]).mean().reset_index()

t1.rename(columns = {"mid":"http_values_count"},inplace = True)
t2.rename(columns = {"forward_count":"http_values_forward_mean"},inplace = True)
t3.rename(columns = {"comment_count":"http_values_comment_mean"},inplace = True)
t4.rename(columns = {"like_count":"http_values_like_mean"},inplace = True)

train1 = pd.merge(train1,t1,how = "left",on = ["http_values"])
train1 = pd.merge(train1,t2,how = "left",on = ["http_values"])
train1 = pd.merge(train1,t3,how = "left",on = ["http_values"])
train1 = pd.merge(train1,t4,how = "left",on = ["http_values"])

test1 = pd.merge(test1,t1,how = "left",on = ["http_values"])
test1 = pd.merge(test1,t2,how = "left",on = ["http_values"])
test1 = pd.merge(test1,t3,how = "left",on = ["http_values"])
test1 = pd.merge(test1,t4,how = "left",on = ["http_values"])

#Processing "uid" of data by train3
#mean of forward_count group by uid
uid_feature = train3["uid"].to_frame()
uid_mean_forward = train3.groupby("uid")["forward_count"].mean().to_frame()
 
#mean of comment_count group by uid
uid_mean_comment = train3.groupby("uid")["comment_count"].mean().to_frame()
 
#mean of like_count group by uid
uid_mean_like = train3.groupby("uid")["like_count"].mean().to_frame()
 
#all uid
uid_count = train3.groupby("uid")["mid"].count().to_frame()
 
#merge
uid_feature = pd.merge(uid_feature,uid_mean_forward,left_on = "uid",right_index = True)
uid_feature = pd.merge(uid_feature,uid_mean_comment,left_on = "uid",right_index = True)
uid_feature = pd.merge(uid_feature,uid_mean_like,left_on = "uid",right_index = True)
train3_uid_feature = pd.merge(uid_feature,uid_count,left_on = "uid",right_index = True)
train3_uid_feature.columns = ["uid","uid_mean_forward","uid_mean_comment","uid_mean_like","uid_count"]

#Processing "uid" of data by test3
#mean of forward_count group by uid
uid_feature = test3["uid"].to_frame()
#merge
uid_feature = pd.merge(uid_feature,uid_mean_forward,left_on = "uid",right_index = True)
uid_feature = pd.merge(uid_feature,uid_mean_comment,left_on = "uid",right_index = True)
uid_feature = pd.merge(uid_feature,uid_mean_like,left_on = "uid",right_index = True)
test3_uid_feature = pd.merge(uid_feature,uid_count,left_on = "uid",right_index = True)
test3_uid_feature.columns = ["uid","uid_mean_forward","uid_mean_comment","uid_mean_like","uid_count"]

#train3 hot_topic_mean
hot_topic_forward_mean = train3["forward_count"].groupby(hot_topic).mean().to_frame().reset_index()
hot_topic_comment_mean = train3["comment_count"].groupby(hot_topic).mean().to_frame().reset_index()
hot_topic_like_mean = train3["like_count"].groupby(hot_topic).mean().to_frame().reset_index()
hot_topic_count = train3["mid"].groupby(hot_topic).count().to_frame().reset_index()
hot_topic_forward_mean.rename(columns = {"forward_count":"hot_topic_forward_mean"},inplace = True)
hot_topic_comment_mean.rename(columns = {"comment_count":"hot_topic_comment_mean"},inplace = True)
hot_topic_like_mean.rename(columns = {"like_count":"hot_topic_like_mean"},inplace = True)
hot_topic_count.rename(columns = {"mid":"hot_topic_count"},inplace = True)
train3 = pd.merge(train3,hot_topic_forward_mean,how = "left",on = "hot_topic")
train3 = pd.merge(train3,hot_topic_comment_mean,how = "left",on = "hot_topic")
train3 = pd.merge(train3,hot_topic_like_mean,how = "left",on = "hot_topic")
train3 = pd.merge(train3,hot_topic_count,how = "left",on = "hot_topic")

#test3 hot_topic_mean
test3 = pd.merge(test3,hot_topic_forward_mean,how = "left",on = "hot_topic")
test3 = pd.merge(test3,hot_topic_comment_mean,how = "left",on = "hot_topic")
test3 = pd.merge(test3,hot_topic_like_mean,how = "left",on = "hot_topic")
test3 = pd.merge(test3,hot_topic_count,how = "left",on = "hot_topic")

#train3 uid_hot_topic_mean
t1 = train3[["uid","hot_topic","mid"]].groupby(["uid","hot_topic"]).count().reset_index()
t2 = train3[["uid","hot_topic","forward_count"]].groupby(["uid","hot_topic"]).mean().reset_index()
t3 = train3[["uid","hot_topic","comment_count"]].groupby(["uid","hot_topic"]).mean().reset_index()
t4 = train3[["uid","hot_topic","like_count"]].groupby(["uid","hot_topic"]).mean().reset_index()

t1.rename(columns = {"mid":"uid_hot_topic_count"},inplace = True)
t2.rename(columns = {"forward_count":"uid_hot_topic_forward_mean"},inplace = True)
t3.rename(columns = {"comment_count":"uid_hot_topic_comment_mean"},inplace = True)
t4.rename(columns = {"like_count":"uid_hot_topic_like_mean"},inplace = True)

train3 = pd.merge(train3,t1,how = "left",on = ["uid","hot_topic"])
train3 = pd.merge(train3,t2,how = "left",on = ["uid","hot_topic"])
train3 = pd.merge(train3,t3,how = "left",on = ["uid","hot_topic"])
train3 = pd.merge(train3,t4,how = "left",on = ["uid","hot_topic"])

#test3 uid_hot_topic_mean
test3 = pd.merge(test3,t1,how = "left",on = ["uid","hot_topic"])
test3 = pd.merge(test3,t2,how = "left",on = ["uid","hot_topic"])
test3 = pd.merge(test3,t3,how = "left",on = ["uid","hot_topic"])
test3 = pd.merge(test3,t4,how = "left",on = ["uid","hot_topic"])

#train3 uid_hour
t1 = train3[["uid","hour","mid"]].groupby(["uid","hour"]).count().reset_index()
t2 = train3[["uid","hour","forward_count"]].groupby(["uid","hour"]).mean().reset_index()
t3 = train3[["uid","hour","comment_count"]].groupby(["uid","hour"]).mean().reset_index()
t4 = train3[["uid","hour","like_count"]].groupby(["uid","hour"]).mean().reset_index()

t1.rename(columns = {"mid":"uid_hour_count"},inplace = True)
t2.rename(columns = {"forward_count":"uid_hour_forward_mean"},inplace = True)
t3.rename(columns = {"comment_count":"uid_hour_comment_mean"},inplace = True)
t4.rename(columns = {"like_count":"uid_hour_like_mean"},inplace = True)

train3 = pd.merge(train3,t1,how = "left",on = ["uid","hour"])
train3 = pd.merge(train3,t2,how = "left",on = ["uid","hour"])
train3 = pd.merge(train3,t3,how = "left",on = ["uid","hour"])
train3 = pd.merge(train3,t4,how = "left",on = ["uid","hour"])

#train3 uid_hour
test3 = pd.merge(test3,t1,how = "left",on = ["uid","hour"])
test3 = pd.merge(test3,t2,how = "left",on = ["uid","hour"])
test3 = pd.merge(test3,t3,how = "left",on = ["uid","hour"])
test3 = pd.merge(test3,t4,how = "left",on = ["uid","hour"])

t1 = train3[["http_values","mid"]].groupby(["http_values"]).count().reset_index()
t2 = train3[["http_values","forward_count"]].groupby(["http_values"]).mean().reset_index()
t3 = train3[["http_values","comment_count"]].groupby(["http_values"]).mean().reset_index()
t4 = train3[["http_values","like_count"]].groupby(["http_values"]).mean().reset_index()

t1.rename(columns = {"mid":"http_values_count"},inplace = True)
t2.rename(columns = {"forward_count":"http_values_forward_mean"},inplace = True)
t3.rename(columns = {"comment_count":"http_values_comment_mean"},inplace = True)
t4.rename(columns = {"like_count":"http_values_like_mean"},inplace = True)

train3 = pd.merge(train3,t1,how = "left",on = ["http_values"])
train3 = pd.merge(train3,t2,how = "left",on = ["http_values"])
train3 = pd.merge(train3,t3,how = "left",on = ["http_values"])
train3 = pd.merge(train3,t4,how = "left",on = ["http_values"])

test3 = pd.merge(test3,t1,how = "left",on = ["http_values"])
test3 = pd.merge(test3,t2,how = "left",on = ["http_values"])
test3 = pd.merge(test3,t3,how = "left",on = ["http_values"])
test3 = pd.merge(test3,t4,how = "left",on = ["http_values"])

t1 = train3[["at_name","like_count"]].groupby("at_name").mean()["like_count"]
t2 = train3[["at_name","forward_count"]].groupby("at_name").mean()["forward_count"]
t3 = train3[["at_name","comment_count"]].groupby("at_name").mean()["comment_count"]
t4 = train3[["at_name","like_count"]].groupby("at_name").count()["like_count"]

train3["at_name_like_count"] = train3["at_name"].map(t1)
train3["at_name_forward_count"] = train3["at_name"].map(t2)
train3["at_name_comment_count"] = train3["at_name"].map(t3)
train3["at_name_count"] = train3["at_name"].map(t4)

test3["at_name_like_count"] = test3["at_name"].map(t1)
test3["at_name_forward_count"] = test3["at_name"].map(t2)
test3["at_name_comment_count"] = test3["at_name"].map(t3)
test3["at_name_count"] = test3["at_name"].map(t4)

#Processing "uid" of data by train2
#mean of forward_count group by uid
uid_feature = train2["uid"].to_frame()
uid_mean_forward = train2.groupby("uid")["forward_count"].mean().to_frame()
 
#mean of comment_count group by uid
uid_mean_comment = train2.groupby("uid")["comment_count"].mean().to_frame()
 
#mean of like_count group by uid
uid_mean_like = train2.groupby("uid")["like_count"].mean().to_frame()
 
#all uid
uid_count = train2.groupby("uid")["mid"].count().to_frame()
 
#merge
uid_feature = pd.merge(uid_feature,uid_mean_forward,left_on = "uid",right_index = True)
uid_feature = pd.merge(uid_feature,uid_mean_comment,left_on = "uid",right_index = True)
uid_feature = pd.merge(uid_feature,uid_mean_like,left_on = "uid",right_index = True)
train2_uid_feature = pd.merge(uid_feature,uid_count,left_on = "uid",right_index = True)
train2_uid_feature.columns = ["uid","uid_mean_forward","uid_mean_comment","uid_mean_like","uid_count"]

#Processing "uid" of data by test2
#mean of forward_count group by uid
uid_feature = test2["uid"].to_frame()
#merge
uid_feature = pd.merge(uid_feature,uid_mean_forward,left_on = "uid",right_index = True)
uid_feature = pd.merge(uid_feature,uid_mean_comment,left_on = "uid",right_index = True)
uid_feature = pd.merge(uid_feature,uid_mean_like,left_on = "uid",right_index = True)
test2_uid_feature = pd.merge(uid_feature,uid_count,left_on = "uid",right_index = True)
test2_uid_feature.columns = ["uid","uid_mean_forward","uid_mean_comment","uid_mean_like","uid_count"]

#train2 hot_topic_mean
hot_topic_forward_mean = train2["forward_count"].groupby(hot_topic).mean().to_frame().reset_index()
hot_topic_comment_mean = train2["comment_count"].groupby(hot_topic).mean().to_frame().reset_index()
hot_topic_like_mean = train2["like_count"].groupby(hot_topic).mean().to_frame().reset_index()
hot_topic_count = train2["mid"].groupby(hot_topic).count().to_frame().reset_index()
hot_topic_forward_mean.rename(columns = {"forward_count":"hot_topic_forward_mean"},inplace = True)
hot_topic_comment_mean.rename(columns = {"comment_count":"hot_topic_comment_mean"},inplace = True)
hot_topic_like_mean.rename(columns = {"like_count":"hot_topic_like_mean"},inplace = True)
hot_topic_count.rename(columns = {"mid":"hot_topic_count"},inplace = True)
train2 = pd.merge(train2,hot_topic_forward_mean,how = "left",on = "hot_topic")
train2 = pd.merge(train2,hot_topic_comment_mean,how = "left",on = "hot_topic")
train2 = pd.merge(train2,hot_topic_like_mean,how = "left",on = "hot_topic")
train2 = pd.merge(train2,hot_topic_count,how = "left",on = "hot_topic")

#test2 hot_topic_mean
test2 = pd.merge(test2,hot_topic_forward_mean,how = "left",on = "hot_topic")
test2 = pd.merge(test2,hot_topic_comment_mean,how = "left",on = "hot_topic")
test2 = pd.merge(test2,hot_topic_like_mean,how = "left",on = "hot_topic")
test2 = pd.merge(test2,hot_topic_count,how = "left",on = "hot_topic")

#train2 uid_hot_topic_mean
t1 = train2[["uid","hot_topic","mid"]].groupby(["uid","hot_topic"]).count().reset_index()
t2 = train2[["uid","hot_topic","forward_count"]].groupby(["uid","hot_topic"]).mean().reset_index()
t3 = train2[["uid","hot_topic","comment_count"]].groupby(["uid","hot_topic"]).mean().reset_index()
t4 = train2[["uid","hot_topic","like_count"]].groupby(["uid","hot_topic"]).mean().reset_index()

t1.rename(columns = {"mid":"uid_hot_topic_count"},inplace = True)
t2.rename(columns = {"forward_count":"uid_hot_topic_forward_mean"},inplace = True)
t3.rename(columns = {"comment_count":"uid_hot_topic_comment_mean"},inplace = True)
t4.rename(columns = {"like_count":"uid_hot_topic_like_mean"},inplace = True)

train2 = pd.merge(train2,t1,how = "left",on = ["uid","hot_topic"])
train2 = pd.merge(train2,t2,how = "left",on = ["uid","hot_topic"])
train2 = pd.merge(train2,t3,how = "left",on = ["uid","hot_topic"])
train2 = pd.merge(train2,t4,how = "left",on = ["uid","hot_topic"])

test2 = pd.merge(test2,t1,how = "left",on = ["uid","hot_topic"])
test2 = pd.merge(test2,t2,how = "left",on = ["uid","hot_topic"])
test2 = pd.merge(test2,t3,how = "left",on = ["uid","hot_topic"])
test2 = pd.merge(test2,t4,how = "left",on = ["uid","hot_topic"])

#train2 uid_hour
t1 = train2[["uid","hour","mid"]].groupby(["uid","hour"]).count().reset_index()
t2 = train2[["uid","hour","forward_count"]].groupby(["uid","hour"]).mean().reset_index()
t3 = train2[["uid","hour","comment_count"]].groupby(["uid","hour"]).mean().reset_index()
t4 = train2[["uid","hour","like_count"]].groupby(["uid","hour"]).mean().reset_index()

t1.rename(columns = {"mid":"uid_hour_count"},inplace = True)
t2.rename(columns = {"forward_count":"uid_hour_forward_mean"},inplace = True)
t3.rename(columns = {"comment_count":"uid_hour_comment_mean"},inplace = True)
t4.rename(columns = {"like_count":"uid_hour_like_mean"},inplace = True)

train2 = pd.merge(train2,t1,how = "left",on = ["uid","hour"])
train2 = pd.merge(train2,t2,how = "left",on = ["uid","hour"])
train2 = pd.merge(train2,t3,how = "left",on = ["uid","hour"])
train2 = pd.merge(train2,t4,how = "left",on = ["uid","hour"])

#test2 uid_hour
test2 = pd.merge(test2,t1,how = "left",on = ["uid","hour"])
test2 = pd.merge(test2,t2,how = "left",on = ["uid","hour"])
test2 = pd.merge(test2,t3,how = "left",on = ["uid","hour"])
test2 = pd.merge(test2,t4,how = "left",on = ["uid","hour"])

t1 = train2[["http_values","mid"]].groupby(["http_values"]).count().reset_index()
t2 = train2[["http_values","forward_count"]].groupby(["http_values"]).mean().reset_index()
t3 = train2[["http_values","comment_count"]].groupby(["http_values"]).mean().reset_index()
t4 = train2[["http_values","like_count"]].groupby(["http_values"]).mean().reset_index()

t1.rename(columns = {"mid":"http_values_count"},inplace = True)
t2.rename(columns = {"forward_count":"http_values_forward_mean"},inplace = True)
t3.rename(columns = {"comment_count":"http_values_comment_mean"},inplace = True)
t4.rename(columns = {"like_count":"http_values_like_mean"},inplace = True)

train2 = pd.merge(train2,t1,how = "left",on = ["http_values"])
train2 = pd.merge(train2,t2,how = "left",on = ["http_values"])
train2 = pd.merge(train2,t3,how = "left",on = ["http_values"])
train2 = pd.merge(train2,t4,how = "left",on = ["http_values"])

test2 = pd.merge(test2,t1,how = "left",on = ["http_values"])
test2 = pd.merge(test2,t2,how = "left",on = ["http_values"])
test2 = pd.merge(test2,t3,how = "left",on = ["http_values"])
test2 = pd.merge(test2,t4,how = "left",on = ["http_values"])

t1 = train2[["at_name","like_count"]].groupby("at_name").mean()["like_count"]
t2 = train2[["at_name","forward_count"]].groupby("at_name").mean()["forward_count"]
t3 = train2[["at_name","comment_count"]].groupby("at_name").mean()["comment_count"]
t4 = train2[["at_name","like_count"]].groupby("at_name").count()["like_count"]

train2["at_name_like_count"] = train2["at_name"].map(t1)
train2["at_name_forward_count"] = train2["at_name"].map(t2)
train2["at_name_comment_count"] = train2["at_name"].map(t3)
train2["at_name_count"] = train2["at_name"].map(t4)

test2["at_name_like_count"] = test2["at_name"].map(t1)
test2["at_name_forward_count"] = test2["at_name"].map(t2)
test2["at_name_comment_count"] = test2["at_name"].map(t3)
test2["at_name_count"] = test2["at_name"].map(t4)

#Processing "content" of data by extract_feature_data train1
#weibo_train_data["content"] = weibo_train_data["content"].astype("str")
 
#content contains http
content_con_http = train1["content"].str.contains("http")
#length of content
content_len = train1["content"].map(lambda x:len(x) if isinstance(x, str) else -1)
#content contains count @
content_count_at = train1["content"].map(lambda x:x.count("@") if isinstance(x, str) else 0)
#content contains @
content_con_at = train1["content"].str.contains("@")
#content count count !
content_count_exc = train1["content"].map(lambda x:x.count("!") if isinstance(x, str) else 0)
#content contains @
content_con_exc = train1["content"].str.contains("!")

#merge
train1_content_feature = pd.concat([content_con_http,content_len,content_count_at,content_con_at,content_count_exc,content_con_exc],axis=1)
train1_content_feature.columns = ["content_con_http","content_len","content_count_at","content_con_at","content_count_exc","content_con_exc"]

#Processing "content" of data by extract_feature_data train2
#weibo_train_data["content"] = weibo_train_data["content"].astype("str")
 
#content contains http
content_con_http = train2["content"].str.contains("http")
#length of content
content_len = train2["content"].map(lambda x:len(x) if isinstance(x, str) else -1)
#content contains count @
content_count_at = train2["content"].map(lambda x:x.count("@") if isinstance(x, str) else 0)
#content contains @
content_con_at = train2["content"].str.contains("@")
#content count count !
content_count_exc = train2["content"].map(lambda x:x.count("!") if isinstance(x, str) else 0)
#content contains @
content_con_exc = train2["content"].str.contains("!")
 
#merge
train2_content_feature = pd.concat([content_con_http,content_len,content_count_at,content_con_at,content_count_exc,content_con_exc],axis=1)
train2_content_feature.columns = ["content_con_http","content_len","content_count_at","content_con_at","content_count_exc","content_con_exc"]

#Processing "content" of data by extract_feature_data train3
#weibo_train_data["content"] = weibo_train_data["content"].astype("str")
 
#content contains http
content_con_http = train3["content"].str.contains("http")
#length of content
content_len = train3["content"].map(lambda x:len(x) if isinstance(x, str) else -1)
#content contains count @
content_count_at = train3["content"].map(lambda x:x.count("@") if isinstance(x, str) else 0)
#content contains @
content_con_at = train3["content"].str.contains("@")
#content count count !
content_count_exc = train3["content"].map(lambda x:x.count("!") if isinstance(x, str) else 0)
#content contains @
content_con_exc = train3["content"].str.contains("!")
 
#merge
train3_content_feature = pd.concat([content_con_http,content_len,content_count_at,content_con_at,content_count_exc,content_con_exc],axis=1)
train3_content_feature.columns = ["content_con_http","content_len","content_count_at","content_con_at","content_count_exc","content_con_exc"]

#Processing "content" of data by extract_feature_data test1
#weibo_train_data["content"] = weibo_train_data["content"].astype("str")
 
#content contains http
content_con_http = test1["content"].str.contains("http")
#length of content
content_len = test1["content"].map(lambda x:len(x) if isinstance(x, str) else -1)
#content contains count @
content_count_at = test1["content"].map(lambda x:x.count("@") if isinstance(x, str) else 0)
#content contains @
content_con_at = test1["content"].str.contains("@")
#content count count !
content_count_exc = test1["content"].map(lambda x:x.count("!") if isinstance(x, str) else 0)
#content contains @
content_con_exc = test1["content"].str.contains("!")
 
#merge
test1_content_feature = pd.concat([content_con_http,content_len,content_count_at,content_con_at,content_count_exc,content_con_exc],axis=1)
test1_content_feature.columns = ["content_con_http","content_len","content_count_at","content_con_at","content_count_exc","content_con_exc"]

#Processing "content" of data by extract_feature_data test1
#weibo_train_data["content"] = weibo_train_data["content"].astype("str")
 
#content contains http
content_con_http = test2["content"].str.contains("http")
#length of content
content_len = test2["content"].map(lambda x:len(x) if isinstance(x, str) else -1)
#content contains count @
content_count_at = test2["content"].map(lambda x:x.count("@") if isinstance(x, str) else 0)
#content contains @
content_con_at = test2["content"].str.contains("@")
#content count count !
content_count_exc = test2["content"].map(lambda x:x.count("!") if isinstance(x, str) else 0)
#content contains @
content_con_exc = test2["content"].str.contains("!")
 
#merge
test2_content_feature = pd.concat([content_con_http,content_len,content_count_at,content_con_at,content_count_exc,content_con_exc],axis=1)
test2_content_feature.columns = ["content_con_http","content_len","content_count_at","content_con_at","content_count_exc","content_con_exc"]

#Processing "content" of data by extract_feature_data test1
#weibo_train_data["content"] = weibo_train_data["content"].astype("str")
 
#content contains http
content_con_http = test2["content"].str.contains("http")
#length of content
content_len = test2["content"].map(lambda x:len(x) if isinstance(x, str) else -1)
#content contains count @
content_count_at = test2["content"].map(lambda x:x.count("@") if isinstance(x, str) else 0)
#content contains @
content_con_at = test2["content"].str.contains("@")
#content count count !
content_count_exc = test2["content"].map(lambda x:x.count("!") if isinstance(x, str) else 0)
#content contains @
content_con_exc = test2["content"].str.contains("!")
 
#merge
test2_content_feature = pd.concat([content_con_http,content_len,content_count_at,content_con_at,content_count_exc,content_con_exc],axis=1)
test2_content_feature.columns = ["content_con_http","content_len","content_count_at","content_con_at","content_count_exc","content_con_exc"]

#Processing "content" of data by extract_feature_data test1
#weibo_train_data["content"] = weibo_train_data["content"].astype("str")
 
#content contains http
content_con_http = test3["content"].str.contains("http")
#length of content
content_len = test3["content"].map(lambda x:len(x) if isinstance(x, str) else -1)
#content contains count @
content_count_at = test3["content"].map(lambda x:x.count("@") if isinstance(x, str) else 0)
#content contains @
content_con_at = test3["content"].str.contains("@")
#content count count !
content_count_exc = test3["content"].map(lambda x:x.count("!") if isinstance(x, str) else 0)
#content contains @
content_con_exc = test3["content"].str.contains("!")
 
#merge
test3_content_feature = pd.concat([content_con_http,content_len,content_count_at,content_con_at,content_count_exc,content_con_exc],axis=1)
test3_content_feature.columns = ["content_con_http","content_len","content_count_at","content_con_at","content_count_exc","content_con_exc"]
del content_con_http
del content_len
del content_count_at
del content_con_at
del content_count_exc
del content_con_exc

train1 = pd.merge(train1,train1_uid_feature,right_index = True,left_on = "uid", how="left")
train1 = pd.concat([train1,train1_content_feature],axis = 1)

train2 = pd.merge(train2,train2_uid_feature,right_index = True,left_on = "uid", how="left")
train2 = pd.concat([train2,train2_content_feature],axis = 1)

train3 = pd.merge(train3,train3_uid_feature,right_index = True,left_on = "uid", how="left")
train3 = pd.concat([train3,train3_content_feature],axis = 1)

test1 = pd.merge(test1,test1_uid_feature,right_index = True,left_on = "uid", how="left")
test1 = pd.concat([test1,test1_content_feature],axis = 1)

test2 = pd.merge(test2,test2_uid_feature,right_index = True,left_on = "uid", how="left")
test2 = pd.concat([test2,test2_content_feature],axis = 1)

test3 = pd.merge(test3,test3_uid_feature,right_index = True,left_on = "uid", how="left")
test3 = pd.concat([test3,test3_content_feature],axis = 1)

#extract_feature =["uid_mean_forward","uid_mean_comment","uid_mean_like","uid_count","content_con_http","content_len","content_count_at","content_con_at","content_count_exc","content_con_exc","hot_topic_forward_mean","hot_topic_comment_mean","hot_topic_like_mean","hot_topic_count","hot_topic_time_delta","hot_topic_first"] 
extract_feature = ["uid_mean_forward","uid_mean_comment","uid_mean_like","uid_count","content_con_http","content_count_at","content_con_at",
                   "content_con_exc","hot_topic_forward_mean","hot_topic_comment_mean","hot_topic_like_mean","hot_topic_count","hot_topic_time_delta",
                   "uid_hot_topic_count","uid_hot_topic_forward_mean","uid_hot_topic_comment_mean","uid_hot_topic_like_mean","weekday",
                   "is_discuss_count","video","hour","uid_hour_count","uid_hour_forward_mean","uid_hour_like_mean","uid_hour_comment_mean",
                   "hot_topic_first","have_title","week_of_year","this_week_hot","next_week_hot","last_week_hot","at_name_like_count",
                   "at_name_forward_count","at_name_comment_count","at_name_count","big_at_name_like_count","big_at_name_forward_count",
                   "big_at_name_comment_count","big_at_name_count","seconds","this_day_hot","next_day_hot","last_day_hot"] 
x_train1 = train1[extract_feature]
y_train1_forward_count = train1["forward_count"]
y_train1_comment_count = train1["comment_count"]
y_train1_like_count = train1["like_count"]

x_train2 = train2[extract_feature]
y_train2_forward_count = train2["forward_count"]
y_train2_comment_count = train2["comment_count"]
y_train2_like_count = train2["like_count"]

x_train3 = train3[extract_feature]
y_train3_forward_count = train3["forward_count"]
y_train3_comment_count = train3["comment_count"]
y_train3_like_count = train3["like_count"]

x_test1 = test1[extract_feature]
y_test1_forward_count = test1["forward_count"]
y_test1_comment_count = test1["comment_count"]
y_test1_like_count = test1["like_count"]

x_test2 = test2[extract_feature]
y_test2_forward_count = test2["forward_count"]
y_test2_comment_count = test2["comment_count"]
y_test2_like_count = test2["like_count"]

x_test3 = test3[extract_feature]

from sklearn.ensemble import RandomForestRegressor as rfc
x_train1 = x_train1.fillna(-1)
x_test1 = x_test1.fillna(-1)

rfc =rfc()
rfc.fit(x_train1,y_train1_forward_count)
predict1_forward_count = rfc.predict(x_test1)
 
rfc.fit(x_train1,y_train1_comment_count)
predict1_comment_count = rfc.predict(x_test1)
 
rfc.fit(x_train1,y_train1_like_count)
predict1_like_count = rfc.predict(x_test1)

x_train1["content_con_http"] = x_train1["content_con_http"].astype(int)
x_train1["content_con_at"] = x_train1["content_con_at"].astype(int)
x_train1["content_con_exc"] = x_train1["content_con_exc"].astype(int)
x_train1["have_title"] = x_train1["have_title"].astype(int)

import xgboost as xgb
model = xgb.XGBRegressor()
model.fit(x_train1,y_train1_forward_count)
predict1_forward_count = rfc.predict(x_test1)
 
model.fit(x_train1,y_train1_comment_count)
predict1_comment_count = rfc.predict(x_test1)
 
model.fit(x_train1,y_train1_like_count)
predict1_like_count = rfc.predict(x_test1)

#0.579578330457
#0.586252710246 -len
#0.589234012995 -is_discuss
#0.606312355787
f = np.abs(y_test1_forward_count - predict1_forward_count)/(y_test1_forward_count+5)
c = np.abs(y_test1_comment_count - predict1_comment_count)/(y_test1_comment_count+3)
l = np.abs(y_test1_like_count - predict1_like_count)/(y_test1_like_count+3)
pre = 1-0.5*f-0.25*c-0.25*l
ci = y_test1_forward_count+y_test1_comment_count+y_test1_like_count
ci[ci>100] = 100

pre = (pre-0.8).map(lambda x:1 if x>0 else 0)
error = ((ci*pre).sum()+pre.sum())/(pre.sum()+len(pre))
print(error)

c = x_train1.columns
s = np.argsort(rfc.feature_importances_)
feature_score = pd.DataFrame(columns=["feature","score"])
feature_score["feature"] = c[s]
feature_score["score"] = rfc.feature_importances_[s]
feature_score

x_train2["content_con_http"] = x_train2["content_con_http"].astype(int)
x_train2["content_con_at"] = x_train2["content_con_at"].astype(int)
x_train2["content_con_exc"] = x_train2["content_con_exc"].astype(int)
x_train2["have_title"] = x_train2["have_title"].astype(int)

model = xgb.XGBRegressor()
rfc.fit(x_train2,y_train2_forward_count)
predict2_forward_count = rfc.predict(x_test2)
 
rfc.fit(x_train2,y_train2_comment_count)
predict2_comment_count = rfc.predict(x_test2)
    
rfc.fit(x_train2,y_train2_like_count)
predict2_like_count = rfc.predict(x_test2)

#0.343881125611
#0.344975067127
#+is_discuss_count 0.359944547485
#+hour 0.506044350346
# uid_hour 0.586583452211
# 0.586252710246 -len
# 0.589898775018 -is_discuss
#0.604538904899
f = np.abs(y_test2_forward_count - predict2_forward_count)/(y_test2_forward_count+5)
c = np.abs(y_test2_comment_count - predict2_comment_count)/(y_test2_comment_count+3)
l = np.abs(y_test2_like_count - predict2_like_count)/(y_test2_like_count+3)
pre = 1-0.5*f-0.25*c-0.25*l
ci = y_test2_forward_count+y_test2_comment_count+y_test2_like_count
ci[ci>100] = 100

pre = (pre-0.8).map(lambda x:1 if x>0 else 0)
error = ((ci*pre).sum()+pre.sum())/(pre.sum()+len(pre))
#print(f,c,l)
print(error)

c = x_train1.columns
s = np.argsort(rfc.feature_importances_)
feature_score = pd.DataFrame(columns=["feature","score"])
feature_score["feature"] = c[s]
feature_score["score"] = rfc.feature_importances_[s]
feature_score

from sklearn.ensemble import RandomForestRegressor as rfc

x_train3 = x_train3.fillna(-1)
x_test3 = x_test3.fillna(-1)

rfc =rfc()
rfc.fit(x_train3,y_train3_forward_count)
predict3_forward_count = rfc.predict(x_test3)
 
rfc.fit(x_train3,y_train3_comment_count)
predict3_comment_count = rfc.predict(x_test3)
 
rfc.fit(x_train3,y_train3_like_count)
predict3_like_count = rfc.predict(x_test3)

c = x_train3.columns
s = np.argsort(rfc.feature_importances_)
feature_score = pd.DataFrame(columns=["feature","score"])
feature_score["feature"] = c[s]
feature_score["score"] = rfc.feature_importances_[s]
feature_score

result = pd.DataFrame(columns=["uid","mid","forward_count","comment_count","like_count"])
test = pd.read_csv("datalab/6336/weibo_predict_data.csv",sep = "\t",parse_dates=['time'],date_parser = dateparse,names = ['uid','mid','time','content'])
result["uid"] = test["uid"]
result["mid"] = test["mid"]
result["forward_count"] = predict3_forward_count
result["comment_count"] = predict3_comment_count
result["like_count"] = predict3_like_count
