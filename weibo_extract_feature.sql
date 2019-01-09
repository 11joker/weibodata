#提取hot_topic,转换time,是否存在http
CREATE TABLE IF NOT EXISTS new_train_data AS
(SELECT uid,mid,forward_count,comment_count,like_count,STR_TO_DATE(time,"%Y-%m-%d %T") as time,
WEEKOFYEAR(time) AS weekofyear,DAY(time) AS days,HOUR(time) AS hours
CASE 
	WHEN INSTR(content,'#')=0 THEN
		NULL
	ELSE
		SUBSTRING_INDEX(SUBSTRING(content,INSTR(content,'#')+1),'#',1)
END AS hot_topic,
CASE 
	WHEN INSTR(content,'http://')!=0 THEN
		1
	ELSE
		0
END AS content_con_http
FROM weibo_train_data); 

#create hot_topic table
CREATE TABLE IF NOT EXISTS hot_topic_table AS
SELECT hot_topic,count(mid) as hot_topic_count,AVG(forward_count) as hot_topic_forward_mean,
AVG(comment_count) as hot_topic_comment_mean,AVG(like_count) as hot_topic_like_mean,MIN(time) AS hot_topic_min_time
FROM new_train_data GROUP BY hot_topic;

#create uid_hot_topic table
CREATE TABLE IF NOT EXISTS uid_hot_topic_table AS
SELECT hot_topic,count(mid) uid_hot_topic_count,AVG(forward_count) as uid_hot_topic_forward_mean,
AVG(comment_count) as uid_hot_topic_comment_mean,AVG(like_count) as uid_hot_topic_like_mean
FROM new_train_data GROUP BY uid,hot_topic;

#create uid_hour table
CREATE TABLE IF NOT EXISTS uid_hour_table AS
SELECT hot_topic,count(mid) uid_hour_count,AVG(forward_count) as uid_hour_forward_mean,
AVG(comment_count) as uid_hour_comment_mean,AVG(like_count) as uid_hour_like_mean
FROM new_train_data GROUP BY uid,HOUR(time);