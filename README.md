# tfg
repo del tfg
En este documento recopilaré todas soluciones a problemas desarrolando mi tfg


## Datasets

g++ -g traversecsv.cpp  -o traversecsv -lmysqlcppconn

después de cada reinicio, entrar a mysql (mysql -u root -p) contraseña: root
SET GLOBAL innodb_buffer_pool_size = 8 * 1024 * 1024 * 1024;


Solución al problema de que se cortasen los inserts a medias:

SELECT @@global.time_zone, @@session.time_zone;
SET GLOBAL time_zone = '+00:00';
SET SESSION time_zone = '+00:00';


### Eventloop future fixes
en vez de tupla de 3 elementos y ejecutar 3 lambdas, el propio condition podría hacer el fullfill promise y así ahorrar una ejecución de función lambda

g++ -g -Wall -std=c++2a -Ieventloop host.cpp -lpthread -L /usr/lib/x86_64-linux-gnu -lmysqlclient -o host

test: ./host 1 1 1000 2 256

**si da core dumped y funcionaba antes, mira el tamaño de serie que definiste**

## MEMORIA

después de añadir elementos a la bibliografía: 
tools->commands->Biber

y luego ya compilar con pdflatex


g++ -g traversecsv.cpp  -o traversecsv -lmysqlcppconn
g++ -o host host.cpp -L /usr/lib/x86_64-linux-gnu -lmysqlclient -lpthread


