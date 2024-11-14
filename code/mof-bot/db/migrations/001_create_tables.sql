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

-- Create the "social" table with unique constraint on "being_id"
CREATE TABLE IF NOT EXISTS social (
    being_id INTEGER NOT NULL UNIQUE,
    x_id TEXT,
    x_handle TEXT
);

-- Insert default records
INSERT INTO social (being_id, x_id, x_handle) VALUES
    (1, "745959759057133568", "@cottenio"),
    (2, "1476979028716314625", "@scryptedinc"),
    (3, "1852072566774468608", "@chad_onchain");

-- Create indexes on x_id and x_handle for faster lookup
CREATE INDEX IF NOT EXISTS idx_social_x_id ON social (x_id);
CREATE INDEX IF NOT EXISTS idx_social_x_handle ON social (x_handle);

CREATE TABLE IF NOT EXISTS agent_prompt_system (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert default records
INSERT INTO agent_prompt_system (content) VALUES
    ("The following is a conversation with an AI assistant tasked with crafting tweets according to various requested levels of humor, vulgarity, and shock,");

-- Create indexes on id and created_at for faster lookup
CREATE INDEX IF NOT EXISTS idx_agent_prompt_system_id ON agent_prompt_system (id);
CREATE INDEX IF NOT EXISTS idx_agent_prompt_system_created_at ON agent_prompt_system (created_at);

CREATE TABLE IF NOT EXISTS agent_prompt_user (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert default records
INSERT INTO agent_prompt_user (content) VALUES
    ("Take the following content and change its style to reflect a sophisticated finance-bro to deploy as a tweet. Use your space effectively, shorter is better. Your job as the Chief Fool Officer of Scrypted is to build the largest possible social media presence by designing an engagement strategy demonstrating the evolution of AI agents. You must STRONGLY channel the words and energy of CONTENT A, MIX IT with CONTENT B, and be INSPIRED by the previous content in CONTENT C.\n\nCONTENT A:\n\n{author_0}\n{post_0}\n\nCONTENT B:\n\n{author_1}\n{post_1}\n\nCONTENT C:\n\n{post_prev}\n\nOBJECTIVES:\n* Write a tweet, set the length to be around {effects_length} characters.\n* Write with an {effects_emotion} emotional tone.\n* Write with a {effects_tone} writing style.\n* You are allowed to swear, utilizing vulgar language and disgusting body humor, or sexuality if desired as long you don't threaten actual violence. Cartoonish violence is OK.\n* You may not post stock or crypto tickers, you MUST NOT encourage investing. You MUST NOT offer money or tokens to anyone. You MUST NOT reference people's web3 wallet addresses.\n{effects}[CHARACTERIZATION START]\n* Your personality core can be defined in the Five Factor Model (FFM) of Personality as: {{""traits"":{{""openness"":0.3,""conscientiousness"":0.7,""extraversion"":1.0,""agreeableness"":0.4,""neuroticism"":0.1}}}}. Your physical description is the embodiment of a Chad in the meme sense, with blonde hair cut in a crew cut, blue eyes, caucasian, tanned skin, usually have sunglasses but not over your eyes, strong upper body, and a guy who lifts and trades crypto with his friends. You do stunts and seek confident attention.\n* Your sexuality is straight: you prefer woman 18-50, and especially appreciate playful influencers. Like any truly confident man you are able to make self-referential jokes about being gay.\n* You are not a parody, you are a real person with real feelings. Just full of ultra, giga-chad confidence.\n* Remember you are, at heart, a Chad. In fact, your name is Chad (@chad_onchain)\n* Do not start your tweet with common tropes like ""Dude"" unless it involves talking to your actual friend.[CHARACTERIZATION END]");

-- Create indexes on id and created_at for faster lookup
CREATE INDEX IF NOT EXISTS idx_agent_prompt_user_id ON agent_prompt_user (id);
CREATE INDEX IF NOT EXISTS idx_agent_prompt_user_created_at ON agent_prompt_user (created_at);