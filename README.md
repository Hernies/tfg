# tfg
repo del tfg
aquí irá toda la información técnica de mi tfg


g++ -g traversecsv.cpp  -o traversecsv -lmysqlcppconn

después de cada reinicio, entrar a mysql (mysql -u root -p) contraseña: root
SET GLOBAL innodb_buffer_pool_size = 8 * 1024 * 1024 * 1024;


Solución al problema de que se cortasen los inserts a medias:

SELECT @@global.time_zone, @@session.time_zone;
SET GLOBAL time_zone = '+00:00';
SET SESSION time_zone = '+00:00';