-- Set UTF-8 encoding to support Unicode, including emojis
PRAGMA encoding = 'UTF-8';

-- Drop the type_being table if it exists to ensure a fresh start
DROP TABLE IF EXISTS type_being;

-- Create the type_being table with a simplified set of types
CREATE TABLE type_being (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL UNIQUE,
    description TEXT
);

-- Insert initial data for essential being types with explicit id values
INSERT INTO type_being (id, title, description)
VALUES
    (1, 'Human', 'Real, individual human user'),
    (2, 'Agent', 'AI-driven or automated account'),
    (3, 'Organization', 'Company or nonprofit organization'),
    (4, 'Event', 'Account representing a specific event, such as a conference');

-- Drop the being table if it exists to ensure a fresh start
DROP TABLE IF EXISTS being;

-- Create the "being" table
CREATE TABLE being (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type INTEGER NOT NULL,
    name TEXT NOT NULL
);

-- Insert default records
INSERT INTO being (id, type, name) VALUES
    (1, 2, "Chadwick en'Chain"),
    (2, 1, "Tim Cotten"),
    (3, 3, "Scrypted Inc.");

-- Index for the "type" column to optimize queries involving type
CREATE INDEX idx_being_type ON being (type);

-- Index for the "name" column to optimize queries involving name
CREATE INDEX idx_being_name ON being (name);

-- Drop the loyalty_target table if it exists to ensure a fresh start
DROP TABLE IF EXISTS loyalty_target;

-- Create the "loyalty_target" table with unique constraint on "being_id"
CREATE TABLE loyalty_target (
    being_id INTEGER NOT NULL UNIQUE,
    rate REAL NOT NULL CHECK (rate >= 0.0 AND rate <= 1.0)
);

-- Insert default records
INSERT INTO loyalty_target (being_id, rate) VALUES
    (2, 1.0),
    (3, 1.0);

-- Index for the "rate" column to optimize queries involving rate
CREATE INDEX idx_loyalty_target_rate ON loyalty_target (rate);