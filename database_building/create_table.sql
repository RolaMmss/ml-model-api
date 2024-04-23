-- Execute this command to create the tables
-- sqlite3 cars.db < database_building/create_table.sql


-- Create a table named Cars with columns corresponding to the columns in your CSV file.

CREATE TABLE IF NOT EXISTS Cars (
    car_ID INTEGER PRIMARY KEY,
    symboling INTEGER,
    CarName TEXT,
    fueltype TEXT,
    aspiration TEXT,
    doornumber TEXT,
    carbody TEXT,
    drivewheel TEXT,
    enginelocation TEXT,
    wheelbase REAL,    --REAL values are stored as 8-byte floating-point numbers.
    carlength REAL,
    carwidth REAL,
    carheight REAL,
    curbweight INTEGER,
    enginetype TEXT,
    cylindernumber TEXT,
    enginesize INTEGER,
    fuelsystem TEXT,
    boreratio REAL,
    stroke REAL,
    compressionratio REAL,
    horsepower INTEGER,
    peakrpm INTEGER,
    citympg INTEGER,
    highwaympg INTEGER,
    price INTEGER
);
