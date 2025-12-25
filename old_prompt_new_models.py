import os
import time
import json

import pandas as pd
from openai import OpenAI
import anthropic
from dotenv import load_dotenv
from gradio_client import Client
from meta_ai_api import MetaAI
from google import genai

load_dotenv()
# Anthropic Claude 3.5 Sonnet
# OpenAI GPT3.5 Turbo
# OpenAI GPT4
# OpenAI GPT4o
# Deepseek V3
# Gemini 2.0
def openai_pipeline(model_name: str, term_df: pd.DataFrame, output_dir: str):
    if model_name == 'deepseek-chat':
        client = OpenAI(api_key=os.getenv("DEEPSEEK_KEY"), base_url="https://api.deepseek.com")
    else:
        client = OpenAI(api_key=os.getenv("CHATGPT_KEY"))
    
    pred_definitions = []
    pred_definitions_list = []

    file_exists = os.path.exists(os.path.join('experiments', output_dir, f'old-{output_dir}.json'))
    if file_exists: # load the prev progress if exists
        with open(os.path.join('experiments', output_dir, f'old-{output_dir}.json'), "r", encoding="utf-8") as file:
            pred_definitions_list = json.load(file)
            prev_result = pd.DataFrame(pred_definitions_list)
            pred_definitions.extend(prev_result['pred_definition'].to_list())
            print(f"{len(prev_result)} term from previous run")
    
    for i, row in term_df.iterrows():
        print(f'current term: {row['term']}')
        prompt = f'Provide the definitioin of {row['term']} drone {row['type']} from a drone expert perspective.'
        
        if file_exists: # skip terms that are already queried
            if row['term'] in prev_result['term'].to_list():
                print(f"{row['term']} exists, skipped!")
                continue

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                stream=False
            )

            term_def = {
                "term": row['term'],
                "pred_definition": response.choices[0].message.content
            }

            pred_definitions.append(response.choices[0].message.content)
            pred_definitions_list.append(term_def)
            with open(os.path.join('experiments', output_dir, f'old-{output_dir}.json'), "w", encoding="utf-8") as file:
                json.dump(pred_definitions_list, file, indent=4)

        except json.JSONDecodeError: # handle error to save progress
            with open(os.path.join('experiments', output_dir, f'old-{output_dir}.json'), "w", encoding="utf-8") as file:
                json.dump(pred_definitions_list, file, indent=4)
            continue

        time.sleep(20) # to prevent Deepseek thinks that the request is a DoS

    term_df['pred_definition'] = pred_definitions
    if not os.path.exists(os.path.join('experiments', output_dir)):
        os.makedirs(os.path.join('experiments', output_dir))
    term_df.to_excel(os.path.join('experiments', output_dir, f'old-{output_dir}.xlsx'))


def claude_pipeline(term_df: pd.DataFrame, output_dir: str):
    client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_KEY"))
    pred_definitions = []
    for i, row in term_df.iterrows():
        prompt = f'Provide the definitioin of {row['term']} drone {row['type']} from a drone expert perspective.'
        response = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}]
        )
        if (response.content[0].text):
            pred_definitions.append(response.content[0].text)
            # with open(os.path.join('experiments', output_dir, f'old-{output_dir}.txt'), "a") as f:
            #     f.write(f"Term: {row['term']}\nDefinition: {response.content[0].text}")
        else:
            print(f"row['term']: {row['term']}, row['type']: {row['type']}")
    
    term_df['pred_definition'] = pred_definitions
    if not os.path.exists(os.path.join('experiments', output_dir)):
        os.makedirs(os.path.join('experiments', output_dir))
    term_df.to_excel(os.path.join('experiments', output_dir, f'old-{output_dir}.xlsx'))


def qwen_pipeline(term_df: pd.DataFrame, output_dir: str):
    client = Client("Qwen/Qwen2.5-Max-Demo")
    pred_definitions = []
    for i, row in term_df.iterrows():
        prompt = f'Provide the definitioin of {row['term']} drone {row['type']} from a drone expert perspective.'
        result = client.predict(
                query=prompt,
                history=[],
                system="You are a helpful assistant.",
                api_name="/model_chat_1"
        )
        print(result[1][0][1])
        if (result[1][0][1]):
            pred_definitions.append(result[1][0][1])
            # with open(os.path.join('experiments', output_dir, f'old-{output_dir}.txt'), "a") as f:
            #     f.write(f"Term: {row['term']}\nDefinition: {result[1][0][1]}")
        else:
            print(f"row['term']: {row['term']}, row['type']: {row['type']}")

    term_df['pred_definition'] = pred_definitions
    if not os.path.exists(os.path.join('experiments', output_dir)):
        os.makedirs(os.path.join('experiments', output_dir))
    term_df.to_excel(os.path.join('experiments', output_dir, f'old-{output_dir}.xlsx'))


def meta_pipeline(term_df: pd.DataFrame, output_dir: str):
    meta_ai = MetaAI()
    pred_definitions = []
    pred_definitions_list = []
    for i, row in term_df.iterrows():
        prompt = f'Provide the definitioin of {row['term']} drone {row['type']} from a drone expert perspective.'
        response = meta_ai.prompt(message=prompt)
        print(f"Term: {row['term']}\nDefinition: {response['message']}")
        if (response['message']):
            pred_definitions.append(response['message'])
            # with open(os.path.join('experiments', output_dir, f'old-{output_dir}.txt'), "a") as f:
            #     f.write(f"Term: {row['term']}\nDefinition: {result[1][0][1]}")
        else:
            print(f"row['term']: {row['term']}, row['type']: {row['type']}")

    term_df['pred_definition'] = pred_definitions
    if not os.path.exists(os.path.join('experiments', output_dir)):
        os.makedirs(os.path.join('experiments', output_dir))
    term_df.to_excel(os.path.join('experiments', output_dir, f'old-{output_dir}.xlsx'))


def gemini_pipeline(term_df: pd.DataFrame, output_dir: str):
    client = genai.Client(api_key=os.getenv("GEMINI_KEY"))
    
    pred_definitions = []
    pred_definitions_list = []

    file_exists = os.path.exists(os.path.join('experiments', output_dir, f'old-{output_dir}.json'))
    if file_exists: # load the prev progress if exists
        with open(os.path.join('experiments', output_dir, f'old-{output_dir}.json'), "r", encoding="utf-8") as file:
            pred_definitions_list = json.load(file)
            prev_result = pd.DataFrame(pred_definitions_list)
            pred_definitions.extend(prev_result['pred_definition'].to_list())
            print(f"{len(prev_result)} term from previous run")
    
    for i, row in term_df.iterrows():
        print(f'current term: {row['term']}')
        prompt = f'Provide the definitioin of {row['term']} drone {row['type']} from a drone expert perspective.'
        
        if file_exists: # skip terms that are already queried
            if row['term'] in prev_result['term'].to_list():
                print(f"{row['term']} exists, skipped!")
                continue

        try:
            response = client.models.generate_content(model='gemini-2.0-flash-exp', contents=prompt)

            term_def = {
                "term": row['term'],
                "pred_definition": response.text
            }

            pred_definitions.append(response.text)
            pred_definitions_list.append(term_def)
            with open(os.path.join('experiments', output_dir, f'old-{output_dir}.json'), "w", encoding="utf-8") as file:
                json.dump(pred_definitions_list, file, indent=4)

        except json.JSONDecodeError: # handle error to save progress
            with open(os.path.join('experiments', output_dir, f'old-{output_dir}.json'), "w", encoding="utf-8") as file:
                json.dump(pred_definitions_list, file, indent=4)
            continue

        time.sleep(20) # to prevent Gemini thinks that the request is a DoS

    term_df['pred_definition'] = pred_definitions
    if not os.path.exists(os.path.join('experiments', output_dir)):
        os.makedirs(os.path.join('experiments', output_dir))
    term_df.to_excel(os.path.join('experiments', output_dir, f'old-{output_dir}.xlsx'))
def main():
    # Load the ground truth
    dataset = pd.read_excel(os.path.join('dataset', 'definition.xlsx'), index_col=0)
    # print(dataset)
    # openai_pipeline('gpt-3.5-turbo-0125', dataset, 'gpt3.5')
    # openai_pipeline('deepseek-chat', dataset, 'deepseek')
    # openai_pipeline('gpt-4', dataset, 'gpt4')
    # openai_pipeline('chatgpt-4o-latest', dataset, 'gpt4o')
    # claude_pipeline(dataset, 'claude-sonnet')
    # qwen_pipeline(dataset, 'qwen')
    meta_pipeline(dataset, 'meta-ai')
    # gemini_pipeline(dataset, 'gemini')


if __name__ == "__main__":
    main()