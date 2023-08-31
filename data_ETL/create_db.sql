DROP TABLE IF EXISTS building_permits;
DROP TABLE IF EXISTS housing_inventory;
DROP TABLE IF EXISTS mortgage_rates;
DROP TABLE IF EXISTS prime_rates;
DROP TABLE IF EXISTS revolving_credit;
DROP TABLE IF EXISTS overall;

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS building_permits
(
    row_id INTEGER PRIMARY KEY autoincrement,
    year_month  INTEGER,
    cbsa_code   INTEGER,
    total_units INTEGER
);

CREATE TABLE IF NOT EXISTS housing_inventory
(
    row_id              INTEGER PRIMARY KEY autoincrement,
    year_month          INTEGER,
    cbsa_code           INTEGER,
    cbsa_title          TEXT,
    total_listing_count INTEGER
);

CREATE TABLE IF NOT EXISTS mortgage_rates
(
    row_id        INTEGER PRIMARY KEY autoincrement,
    year_month    INTEGER,
    mortgage_rate NUMERIC
);

CREATE TABLE IF NOT EXISTS prime_rates
(
    row_id     INTEGER PRIMARY KEY autoincrement,
    year_month INTEGER,
    prime_rate NUMERIC
);

CREATE TABLE IF NOT EXISTS revolving_credit
(
    row_id     INTEGER PRIMARY KEY autoincrement,
    year_month INTEGER,
    credit     NUMERIC
);
