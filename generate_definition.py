import os
import openai
import pandas as pd
import numpy as np
import sqlite3
from dotenv import load_dotenv
from setup_db import get_connection
from database.queries import get_term, insert_term, insert_definition

load_dotenv()


def build_item_list(preds_df):
    item_list = []
    term = ""
    label = ""
    prev_type = ""
    for row in range(0, preds_df.shape[0]):
        current_word = preds_df.iloc[row, 0]
        # next_word = preds_df.iloc[row + 1, 0]
        current_label = preds_df.iloc[row, 2]
        current_type = current_label.split('-')[-1]
        # next_label = preds_df.iloc[row + 1, 2]
        if current_label == 'O':
            if term != "":
                print("index: ", row+1)
                item_list.append((term, label))
            term = ""
            label = ""
            continue
        else:
            if current_label.split('-')[0] == "B":
                if term != "":
                    print("index: ", row+1)
                    item_list.append((term, label))
                term = current_word
                label = current_label.split('-')[-1]
            elif prev_type != current_type:
                if term != "":
                    print("index: ", row+1)
                    item_list.append((term, label))
                term = current_word
                label = current_label.split('-')[-1]
            else:
                term = current_word if term == "" else term + " " + current_word
                label = current_label.split('-')[-1]
            prev_type = current_type
    return item_list


def chatgpt_pipeline(db_path, item_list):
    # Load your API key from an environment variable or secret management service
    print("Running ChatGPT pipeline...")
    openai.api_key = os.getenv("CHATGPT_KEY")
    # try:
    connection = get_connection(db_path)

    for term, entity_type in item_list:
        # Check if the term exists in the DB, insert if does not exists
        entity_type = entity_type.lower()
        term_exists = get_term(connection, term, entity_type)
        if term_exists == None:
            insert_term(connection, term, entity_type)

        # Prompt if the definition does not exist
        prompt = "Provide the definition of {} drone {} from a drone expert perspective".format(
            term, entity_type)
        response = openai.Completion.create(
            model="text-davinci-003", prompt=prompt, temperature=0, max_tokens=1000)
        definition = response.choices[0].text.strip()

        # Save the definition into .txt files for backup
        isExist = os.path.exists('definitions/chatgpt')
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs('definitions/chatgpt')
        with open('definitions/chatgpt/{}-{}.txt'.format(term.replace(' ', '_'), entity_type), 'w') as file:
            file.write(definition)

        # Store the definition to the DB
        inserted_definition = insert_definition(
            connection, term, entity_type, 'chatgpt', prompt, definition)
        if isinstance(inserted_definition, sqlite3.Error):
            continue
    print("\nChatGPT Pipeline finished successfully!")
    # except sqlite3.Error as error:
    #     connection.rollback()
    #     print("\nChatGPT Pipeline failed: ", error)


def main():
    # Load your API key from an environment variable or secret management service
    openai.api_key = os.getenv("CHATGPT_KEY")

    pred_df = pd.read_csv(
        "ner_results/bert-base-cased/prediction_bert-base-cased.csv")
    item_list = build_item_list(pred_df)
    db_path = 'database/drone_definitions.db'
    # Run ChatGPT Pipeline to generate definitions from OpenAI ChatGPT
    chatgpt_pipeline(db_path, item_list)
    # terms = [('Obstacle Avoidance', 'function'), ('Palm Control',
    #   'function'), ('Auxiliary Bottom Light', 'component')]
    # for term, entity_type in item_list:
    #     prompt = "Provide the definition of {} drone {} from a drone expert perspective".format(
    #         term, entity_type)
    #     query = [
    #         ['Represent the Wikipedia question for retrieving supporting documents: ', prompt]]
    #     response = openai.Completion.create(
    #         model="text-davinci-003", prompt=prompt, temperature=0, max_tokens=1000)
    #     definition = response.choices[0].text.strip()
    #     with open('definitions/{}/{}.txt'.format('chatgpt', term.replace(' ', '_')), 'w') as file:
    #         file.write(definition)
    #     query_embedding = model.encode(query).reshape(1, -1)
    #     document_embedding = model.encode(definition).reshape(1, -1)
    #     euclidean_distance = euclidean_distances(
    #         query_embedding, document_embedding)
    #     cosine_distance = cosine_distances(query_embedding, document_embedding)
    #     cosine_sim_score = cosine_similarity(
    #         query_embedding, document_embedding)
    #     manhattan_distance = manhattan_distances(
    #         query_embedding, document_embedding)
    #     with open('{}_{}.txt'.format('chatgpt', term), 'w') as file:
    #         file.write(definition + '\n' + 'Euclidean Distance: {}'.format(euclidean_distance) + '\n' +
    #                    'Cosine Distance: {}'.format(cosine_distance) + '\n' + 'Manhattan Distance: {}'.format(manhattan_distance) + '\n' + 'Cosine Similarity: {}'.format(cosine_sim_score))
    #     print(term)
    #     print(entity_type)
    #     print(definition)


if __name__ == "__main__":
    main()
    # pred_df = pd.read_csv(
    #     "ner_results/bert-base-cased/prediction_bert-base-cased.csv")
    # item_list = build_item_list(pred_df)
    # item_df = pd.DataFrame(item_list, columns=['Term', 'Entity Type'])
    # item_df.to_csv('item_list.csv')
    # print(item_df)
    # main()
    # term = "Obstacle Avoidance"
    # entity_type = 'function'
    # prompt = "Provide the definition of {} drone {} from a drone expert perspective".format(
    #     term, entity_type)
    # file = open("definitions/chatgpt/Obstacle_Avoidance.txt", "r")
    # definition = file.read()
    # euclidean_distance, cosine_distance, manhattan_distance = compute_distance(
    #     prompt, definition)
    # print(euclidean_distance, cosine_distance,
    #       manhattan_distance)
