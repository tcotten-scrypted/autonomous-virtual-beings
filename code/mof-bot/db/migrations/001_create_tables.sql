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
