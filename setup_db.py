import sqlite3


def get_connection(db_name):
    conn = sqlite3.connect(db_name)
    return conn


def create_tables(connection):
    cursor = connection.cursor()

    # Create a table for entity_types
    create_table_query = '''CREATE TABLE IF NOT EXISTS entity_types
                            (id INTEGER PRIMARY KEY,
                            name VARCHAR NOT NULL,
                            description TEXT NOT NULL)'''
    cursor.execute(create_table_query)

    # Create a table for terms
    create_table_query = '''CREATE TABLE IF NOT EXISTS terms
                            (id INTEGER PRIMARY KEY,
                            term VARCHAR NOT NULL,
                            entity_type_id INTEGER NOT NULL,
                            entity_type VARCHAR NOT NULL,
                            UNIQUE (term, entity_type_id),
                            FOREIGN KEY (entity_type_id) REFERENCES entity_types (id))'''
    cursor.execute(create_table_query)

    # Create a table for prompt_templates
    create_table_query = '''CREATE TABLE IF NOT EXISTS prompt_templates
                            (id INTEGER PRIMARY KEY,
                            name TEXT NOT NULL,
                            template TEXT NOT NULL,
                            description TEXT NOT NULL)'''
    cursor.execute(create_table_query)

    # Create a table for models
    create_table_query = '''CREATE TABLE IF NOT EXISTS models
                            (id INTEGER PRIMARY KEY,
                            model_name VARCHAR NOT NULL,
                            company VARCHAR NULL,
                            version VARCHAR NULL,
                            size VARCHAR NULL)'''
    cursor.execute(create_table_query)

    # Create a table for attempts
    create_table_query = '''CREATE TABLE IF NOT EXISTS attempts
                            (id INTEGER PRIMARY KEY,
                            term_id INTEGER NOT NULL,
                            model_id INTEGER NOT NULL,
                            counter INTEGER NOT NULL,
                            FOREIGN KEY (term_id) REFERENCES terms (id),
                            FOREIGN KEY (model_id) REFERENCES models (id))'''
    cursor.execute(create_table_query)

    # Create a table for prompts
    create_table_query = '''CREATE TABLE IF NOT EXISTS prompts
                            (id INTEGER PRIMARY KEY,
                            prompt TEXT NOT NULL,
                            term_id INTEGER NOT NULL,
                            term VARCHAR NOT NULL,
                            template_id INTEGER NOT NULL,
                            FOREIGN KEY (term_id) REFERENCES terms (id),
                            FOREIGN KEY (template_id) REFERENCES prompt_templates (id))'''
    cursor.execute(create_table_query)

    # Create a table for definitions
    create_table_query = '''CREATE TABLE IF NOT EXISTS definitions
                            (id INTEGER PRIMARY KEY,
                            definition TEXT NOT NULL,
                            term_id INTEGER NOT NULL,
                            term VARCHAR NOT NULL,
                            model_id INTEGER NOT NULL,
                            model_name VARCHAR NOT NULL,
                            prompt_id INTEGER NOT NULL,
                            prompt TEXT NOT NULL,
                            counter INTEGER NOT NULL,
                            cosine_distance FLOAT NULL,
                            euclidean_distance FLOAT NULL,
                            manhattan_distance FLOAT NULL,
                            sum_distance FLOAT NULL,
                            FOREIGN KEY (term_id) REFERENCES terms (id),
                            FOREIGN KEY (prompt_id) REFERENCES prompts (id),
                            FOREIGN KEY (model_id) REFERENCES models (id))'''
    cursor.execute(create_table_query)
    connection.commit()


def insert_data(connection):
    cursor = connection.cursor()
    # cursor.execute('''INSERT INTO 'entity_types' ('name', 'description') VALUES
    #                 ('Issue', 'Words/phrases that indicate some issues happen to the drone.'),
    #                 ('Parameter', 'Words/phrases that represent some parameters of configuration in a drone.'),
    #                 ('Action', 'Words/phrases that indicate some actions taken by the drone.'),
    #                 ('Component', 'Words/phrases that reflect physical components of a drone.'),
    #                 ('Function', 'Words/phrases that denote some functionalities or features of a drone equipped with.'),
    #                 ('State', 'Words/phrases that notify a state/mode of a drone operates in during a flight.')
    #                 ''')
    # cursor.execute('''INSERT INTO 'prompt_templates' ('name', 'template', 'description') VALUES
    #                 ('persona', 'Provide the definition of term drone entity_type from a drone expert perspective!', 'Prompt template by using persona.'),
    #                 ('query', 'Define term drone entity_type!', 'Prompt template by using query template for retrieval.')
    #                 ''')
    cursor.execute('''INSERT INTO 'models' ('model_name', 'company', 'version', 'size') VALUES
                    ('chatgpt', 'Open AI', 'text-davinci-003', NULL),
                    ('chatsonic', 'Write Sonic', 'premium', NULL)
                    ''')
    connection.commit()


def main():
    connection = get_connection('drone_definitions.db')
    create_tables(connection)
    insert_data(connection)
    connection.close()


if __name__ == "__main__":
    main()
