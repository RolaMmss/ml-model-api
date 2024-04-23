-- Execute this command to create the tables
-- sqlite3 cars.db  < database_building/import_table.sql


.import --csv --skip 1 -v data/carprice_original.csv Cars 



