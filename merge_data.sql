CREATE TABLE extract_feautre_table AS SELECT
new_train_data.*,
hot_topic_table.*,
uid_hot_topic_table.*,
uid_hour_table.*,
(UNIX_TIMESTAMP(new_train_data.time) - UNIX_TIMESTAMP(hot_topic_table.hot_topic_min_time)) AS hot_topic_time_delta
FROM
	new_train_data
	LEFT JOIN hot_topic_table ON new_train_data.hot_topic = hot_topic_table.hot_topic
	LEFT JOIN uid_hot_topic_table ON new_train_data.uid = uid_hot_topic_table.uid 
	AND new_train_data.hot_topic = uid_hot_topic_table.hot_topic
	LEFT JOIN uid_hour_table ON uid_hour_table.hours = new_train_data.hours 
	AND uid_hour_table.uid = new_train_data.uid