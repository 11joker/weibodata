#提取hot_topic,转换time,是否存在http
CREATE TABLE
IF
	NOT EXISTS new_train_data AS (
	SELECT
		uid,
		mid,
		forward_count,
		comment_count,
		like_count,
		STR_TO_DATE( time, "%Y-%m-%d %T" ) AS time,
		WEEKOFYEAR( time ) AS this_week,
		WEEKOFYEAR( time ) + 1 AS last_week,
		WEEKOFYEAR( time ) - 1 AS next_week,
		DATEDIFF(time,MIN(time)) AS days,
		HOUR ( time ) AS hours,
		WEEKDAY(time) AS weekday,
		len(content)-len(REPLACE(content,'@','')) AS content_con_at
		len(content)-len(REPLACE(content,'//',''))-len(content)-len(REPLACE(content,'http://','')) AS is_discuss_count
	CASE
			WHEN INSTR( content, '#' ) = 0 THEN
			NULL ELSE SUBSTRING_INDEX(SUBSTRING( content, INSTR( content, '#' ) + 1 ), '#', 1 ) 
		END AS hot_topic,
	CASE
			WHEN INSTR( content, 'http://' ) != 0 THEN
			1 ELSE 0 
		END AS content_con_http 
	FROM
		weibo_train_data 
	);
	
CREATE TABLE
IF
	NOT EXISTS weekofyear AS ( SELECT this_week, COUNT( mid ) AS this_week_hot FROM new_train_data GROUP BY this_week ) a
	LEFT JOIN ( SELECT next_week, COUNT( mid ) AS next_week_hot FROM new_train_data GROUP BY next_week ) b ON a.this_week = b.next_week
	LEFT JOIN ( SELECT last_week, COUNT( mid ) AS last_week_hot FROM new_train_data GROUP BY last_week ) c ON a.this_week = c.last_week;
ALTER new_train_data DROP last_week,
next_week;

#create hot_topic table
CREATE TABLE
IF
	NOT EXISTS hot_topic_table AS SELECT
	hot_topic,
	count( mid ) AS hot_topic_count,
	AVG( forward_count ) AS hot_topic_forward_mean,
	AVG( comment_count ) AS hot_topic_comment_mean,
	AVG( like_count ) AS hot_topic_like_mean,
	MIN( time ) AS hot_topic_min_time 
FROM
	new_train_data 
GROUP BY
	hot_topic;
	
#create uid_table table
CREATE TABLE
IF
	NOT EXISTS uid_table AS SELECT
	uid,
	count( mid ) AS uid_count,
	AVG( forward_count ) AS uid_forward_mean,
	AVG( comment_count ) AS uid_comment_mean,
	AVG( like_count ) AS uid_like_mean,
	MIN( time ) AS uid_min_time 
FROM
	new_train_data 
GROUP BY
	uid;
	
#create uid_hot_topic table
CREATE TABLE
IF
	NOT EXISTS uid_hot_topic_table AS SELECT
	uid,
	hot_topic,
	count( mid ) uid_hot_topic_count,
	AVG( forward_count ) AS uid_hot_topic_forward_mean,
	AVG( comment_count ) AS uid_hot_topic_comment_mean,
	AVG( like_count ) AS uid_hot_topic_like_mean 
FROM
	new_train_data 
GROUP BY
	uid,
	hot_topic;
	
#create uid_hour table
CREATE TABLE
IF
	NOT EXISTS uid_hour_table AS SELECT
	uid,
	HOUR ( time ) AS hours,
	hot_topic,
	count( mid ) uid_hour_count,
	AVG( forward_count ) AS uid_hour_forward_mean,
	AVG( comment_count ) AS uid_hour_comment_mean,
	AVG( like_count ) AS uid_hour_like_mean 
FROM
	new_train_data 
GROUP BY
	uid,
	HOUR ( time );