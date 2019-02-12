# weibodata
竞赛题目
对于一条原创博文而言,转发、评论、赞等互动行为能够体现出用户对于博文内容的兴趣程度，也是对博文进行分发控制的重要参考指标。本届赛题的任务就是根据抽样用户的原创博文在发表一天后的转发、评论、赞总数，建立博文的互动模型，并预测用户后续博文在发表一天后的互动情况。

数据说明

第一赛季数据
l  训练数据（weibo_train_data(new)）2015-02-01至2015-07-31
博文的全部信息都映射为一行数据。其中对用户做了一定抽样，获取了抽样用户半年的原创博文，对用户标记和博文标记做了加密 发博时间精确到天级别。 

字段

字段说明

提取说明

uid

用户标记

抽样&字段加密

mid

博文标记

抽样&字段加密

time

发博时间

精确到天

forward_count

博文发表一周后的转发数

 
comment_count

博文发表一周后的评论数

 
like_count

博文发表一周后的赞数

 
content

博文内容

 

l  预测数据（weibo_predict_data(new)）2015-08-01至2015-08-31

字段

字段说明

提取说明

uid

用户标记

抽样&字段加密

mid

博文标记

抽样&字段加密

time

发博时间

精确到天

content

博文内容

 

l  选手需要提交的数据（weibo_result_data），选手对预测数据（weibo_predict_data）中每条博文一周后的转、评、赞值进行预测

字段

字段说明

提取说明

uid

用户标记

抽样&字段加密

mid

博文标记

抽样&字段加密

forward_count

博文发表一周后的转发数

 
comment_count

博文发表一周后的评论数

 
like_count

博文发表一周后的赞数

 
选手提交结果文件的转、评、赞值必须为整数不接受浮点数！注意：提交格式(.txt)：uid、mid、forward_count字段以tab键分隔，forward_count、comment_count、like_count字段间以逗号分隔

#-------------------
思路
提取热门话题hot—topic（两个个#号之间）
1.按热门话题分组计算forward_count、comment_count、like_count平均值
2.该博文离自己的热门话题距离时间（随时间热门话题逐渐冷淡）

是否包含http://字符

是否包含@字符

分解时间（几点，星期几，该年第几周）

按用户分组计算forward_count、comment_count、like_count平均值，发布数量

按用户-热点话题分组计算forward_count、comment_count、like_count平均值

按小时分组计算forward_count、comment_count、like_count平均值

提取@的用户，进行分组计算forward_count、comment_count、like_count平均值，个数
