import sqlite3
from utils.metrics import compute_distance


def get_term(connection, term, entity_type_name):
    cursor = connection.cursor()
    query = """SELECT * FROM terms WHERE term='{}' AND entity_type='{}'""".format(
        term, entity_type_name)
    cursor.execute(query)
    row = cursor.fetchone()
    return row


def insert_term(connection, term, entity_type_name):
    entity_type = get_entity_by_name(connection, entity_type_name)
    cursor = connection.cursor()
    query = """INSERT INTO terms ('term', 'entity_type_id', 'entity_type') VALUES
                    ('{}', {}, '{}')""".format(term, entity_type[0], entity_type_name)
    # cursor.execute('''INSERT INTO 'terms' ('term', 'entity_type_id', 'entity_type') VALUES
    #                 ('chatgpt', 'Open AI', 'text-davinci-003', NULL)
    cursor.execute(query)
    connection.commit()
    #                 ''')
    return cursor.lastrowid
    # try:
    # except sqlite3.Error as error:
    #     raise error
    #     print("Failed to insert new term: ", error)


def get_model_by_name(connection, model_name):
    cursor = connection.cursor()
    model_query = """SELECT * FROM models WHERE model_name='{}'""".format(
        model_name)
    cursor.execute(model_query)
    llm_model = cursor.fetchone()
    return llm_model


def get_entity_by_name(connection, name):
    cursor = connection.cursor()
    # print('name: ', name)
    cursor.execute("SELECT * FROM entity_types WHERE name=?", (name,))
    row = cursor.fetchone()
    cursor.close()
    # print('get entity: ',  row)
    return row


def get_prompt_template(connection, name):
    cursor = connection.cursor()
    query = """SELECT * FROM prompt_templates WHERE name='{}'""".format(name)
    cursor.execute(query)
    template = cursor.fetchone()
    return template


def get_prompt(connection, term_id, template_id):
    cursor = connection.cursor()
    query = """SELECT * FROM prompts WHERE term_id = {} AND template_id = {}""".format(
        term_id, template_id)
    cursor.execute(query)
    prompt = cursor.fetchone()
    return prompt


def insert_prompt(connection, prompt, term_id, term, template_id):
    cursor = connection.cursor()
    insert_prompt_query = """INSERT INTO prompts ('prompt', 'term_id', 'term', 'template_id') VALUES
            ('{}', {}, '{}', {})""".format(prompt, term_id, term, template_id)
    cursor.execute(insert_prompt_query)
    connection.commit()
    return cursor.lastrowid
    # try:
    # except sqlite3.Error as error:
    #     raise error
    #     print("Failed to insert new prompt: ", error)


def get_counter(connection, term_id, model_id):
    cursor = connection.cursor()
    query = """SELECT * FROM attempts WHERE term_id={} AND model_id={}""".format(
        term_id, model_id)

    cursor.execute(query)
    counter = cursor.fetchone()
    return counter


def get_definition(connection, term_id, model):
    cursor = connection.cursor()
    query = """SELECT * FROM definitions WHERE term_id={} AND model_name='{}'""".format(
        term_id, model)
    cursor.execute(query)
    row = cursor.fetchone()
    return row


def update_counter(connection, term_id, model_id):
    cursor = connection.cursor()
    attempt = get_counter(term_id, model_id)
    counter = attempt[3] + 1
    update_counter_query = """UPDATE attempts set counter = {} WHERE id = {}""".format(
        counter, attempt[0])
    cursor.execute(update_counter_query)
    connection.commit()
    return counter
    # try:
    # except sqlite3.Error as error:
    #     raise error
    #     print("Failed to update counter: ", error)


def insert_definition(connection, term, entity_type_name, model_name, prompt, definition):
    # term_query = """SELECT * FROM terms WHERE term='{}' AND entity_type='{}'""".format(
    #     term, entity_type_name)
    # cursor.execute(term_query)
    term = get_term(connection, term, entity_type_name)

    # model_query = """SELECT * FROM models WHERE model_name='{}'""".format(
    #     model_name)
    # cursor.execute(model_query)
    llm_model = get_model_by_name(connection, model_name)

    # Check if the prompt exists, insert if does not exist
    cursor = connection.cursor()
    prompt_exists = get_prompt(connection, term[0], 1)
    new_prompt = None
    if prompt_exists == None:
        new_prompt = insert_prompt(connection, prompt, term[0], term[1], 1)
        # insert_prompt_query = """INSERT INTO prompts ('prompt', 'term_id', 'term', 'template_id') VALUES
        # ('{}', '{}', {}, {})""".format(prompt, term[0], term[1], 1)
        # cursor.execute(insert_prompt_query)
        # connection.commit()
        # new_prompt = cursor.lastrowid
    prompt_id = prompt_exists[0] if prompt_exists != None else new_prompt
    definition_exist = get_definition(connection,
                                      term[0], model_name)
    counter = 1
    if definition_exist != None:
        counter = update_counter(connection, term[0], llm_model[0])

    euclidean_distance, cosine_distance, manhattan_distance = compute_distance(
        prompt, definition)
    sum_distance = euclidean_distance + cosine_distance + manhattan_distance
    insert_query = """INSERT INTO definitions ('definition', 'term_id', 'term', 'model_id', 'model_name', 'prompt_id', 'prompt', 'counter', 'cosine_distance', 'euclidean_distance', 'manhattan_distance', 'sum_distance') VALUES
            ('{}', {}, '{}', {}, '{}', {}, '{}', {}, {}, {}, {}, {})
            """.format(definition, term[0], term[1], llm_model[0], model_name, prompt_id, prompt, counter, cosine_distance, euclidean_distance, manhattan_distance)
    print('insert def: ', insert_query)
    cursor.execute(insert_query)
    if cursor.rowcount > 0:
        connection.commit()
        return cursor.lastrowid
    else:
        connection.rollback()
        return False
    # try:
    # except sqlite3.Error as error:
    #     connection.rollback()
    #     print("\nFailed to insert the definition of term {}: {}".format(
    #         term, error))
    #     return error
