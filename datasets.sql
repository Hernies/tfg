insert into House (`idHouse`) values ('House10');
select * from datapoint  where House_idHouse=3 order by iddatapoint asc limit 1024;
select count(*) from datapoint  where House_idHouse=1;
select * from House;

SELECT @@global.time_zone, @@session.time_zone;
SET GLOBAL time_zone = '+00:00';
SET SESSION time_zone = '+00:00';

SET GLOBAL innodb_buffer_pool_size = 8 * 1024 * 1024 * 1024;

SELECT * FROM datapoint WHERE House_idHouse=1 order by iddatapoint asc LIMIT 200 OFFSET 0;



INSERT INTO `datasets`.`datapoint` ( `House_idHouse`, `timestamp`, `aggregate`, `appliance1`, `appliance2`, `appliance3`, `appliance4`, `appliance5`, `appliance6`, `appliance7`, `appliance8`, `appliance9`, `issues`) VALUES (9, '2014-03-30 02:00:14', 394, 222, 0, 0, 0, 1, 0, 0, 9, 0, 0);
SELECT * FROM datasets WHERE House = 1 LIMIT 200 OFFSET 0;
SELECT * FROM datasets.datapoint WHERE House_idHouse=1 order by iddatapoint asc LIMIT 200 OFFSET 0;

show columns from datasets.datapoint
