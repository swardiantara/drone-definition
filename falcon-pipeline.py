from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import os
import sqlite3
import pandas as pd
from utils.main import build_item_list
from setup_db import get_connection
from database.queries import get_term, insert_term, insert_definition, get_counter, get_definition


def falcon_pipeline(db_path, item_list):
    print("Running Falcon pipeline...")
    model = "tiiuae/falcon-7b"

    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    connection = get_connection(db_path)
    for term, entity_type in item_list:
        # Check if the term exists in the DB, insert if does not exists
        entity_type = entity_type.lower()
        term_exists = get_term(connection, term, entity_type)
        inserted_term = None
        if term_exists == None:
            inserted_term = insert_term(connection, term, entity_type)
        term_id = term_exists[0] if term_exists != None else inserted_term

        # Check if the definition exists
        definition_exist = get_definition(connection,
                                          term_id, 'falcon')
        if definition_exist != None:
            continue

        # Prompt if the definition does not exist
        prompt = "Provide the definition of {} drone {} from a drone expert perspective!".format(
            term, entity_type)
        question = "Question: {}\nAnswer: {} drone {} is".format(
            prompt, term, entity_type)
        sequences = pipeline(
            question,
            max_length=500,
            do_sample=True,
            top_k=2,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )

        definition = sequences[0]['generated_text'].split('\n')[1].split(':')[
            1].strip()
        # Save the definition into .txt files for backup
        isExist = os.path.exists('definitions/falcon')
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs('definitions/falcon')
        with open('definitions/falcon/{}-{}.txt'.format(term.replace(' ', '_'), entity_type), 'w') as file:
            file.write(definition)

        # Store the definition to the DB
        inserted_definition = insert_definition(
            connection, term, entity_type, 'falcon', prompt, definition)
        if isinstance(inserted_definition, sqlite3.Error):
            continue
        elif inserted_definition == False:
            continue
    print("\nFalcon Pipeline finished successfully!")


def main():
    pred_df = pd.read_csv(
        "ner_results/bert-base-cased/prediction_bert-base-cased.csv")
    item_list = build_item_list(pred_df)
    db_path = 'database/drone_definitions.db'
    falcon_pipeline(db_path, item_list)


if __name__ == "__main__":
    main()
