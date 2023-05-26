import os
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, cosine_distances, manhattan_distances
from InstructorEmbedding import INSTRUCTOR
from dotenv import load_dotenv

load_dotenv()


def main():
    # Load your API key from an environment variable or secret management service
    model = INSTRUCTOR('hkunlp/instructor-base')
    openai.api_key = os.getenv("CHATGPT_KEY")
    terms = [('Obstacle Avoidance', 'function'), ('Palm Control',
                                                  'function'), ('Auxiliary Bottom Light', 'component')]
    for term, entity_type in terms:
        prompt = "Provide the definition of {} drone {} from a drone expert perspective".format(
            term, entity_type)
        query = [
            ['Represent the Wikipedia question for retrieving supporting documents: ', prompt]]
        response = openai.Completion.create(
            model="text-davinci-003", prompt=prompt, temperature=0, max_tokens=1000)
        definition = response.choices[0].text.strip()
        with open('definitions/{}/{}.txt'.format('chatgpt', term.replace(' ', '_')), 'w') as file:
            file.write(definition)
        query_embedding = model.encode(query).reshape(1, -1)
        document_embedding = model.encode(definition).reshape(1, -1)
        euclidean_distance = euclidean_distances(
            query_embedding, document_embedding)
        cosine_distance = cosine_distances(query_embedding, document_embedding)
        cosine_sim_score = cosine_similarity(
            query_embedding, document_embedding)
        manhattan_distance = manhattan_distances(
            query_embedding, document_embedding)
        with open('{}_{}.txt'.format('chatgpt', term), 'w') as file:
            file.write(definition + '\n' + 'Euclidean Distance: {}'.format(euclidean_distance) + '\n' +
                       'Cosine Distance: {}'.format(cosine_distance) + '\n' + 'Manhattan Distance: {}'.format(manhattan_distance) + '\n' + 'Cosine Similarity: {}'.format(cosine_sim_score))
        print(term)
        print(entity_type)
        print(definition)


def compute_distance(prompt, definition):
    embedder = INSTRUCTOR('hkunlp/instructor-base')
    embedder.max_seq_length = 1024
    print('dimension: ', embedder.get_sentence_embedding_dimension())
    print('config:', embedder._model_config)
    # embedder._first_module().size = 1024
    query_embedding = embedder.encode(prompt)
    document_embedding = embedder.encode(definition)
    print("query", len(query_embedding))
    print("document_embedding", len(document_embedding))
    query_embedding = embedder.encode(prompt).reshape(1, -1)
    document_embedding = embedder.encode(definition).reshape(1, -1)
    print("query", len(query_embedding))
    print("document_embedding", len(document_embedding))
    euclidean_distance = euclidean_distances(
        query_embedding, document_embedding)
    cosine_distance = cosine_distances(query_embedding, document_embedding)
    cosine_sim_score = cosine_similarity(
        query_embedding, document_embedding)
    manhattan_distance = manhattan_distances(
        query_embedding, document_embedding)
    return euclidean_distance, cosine_distance, cosine_sim_score, manhattan_distance


if __name__ == "__main__":
    term = "Obstacle Avoidance"
    entity_type = 'function'
    prompt = "Provide the definition of {} drone {} from a drone expert perspective".format(
        term, entity_type)
    file = open("definitions/chatgpt/Obstacle_Avoidance.txt", "r")
    definition = file.read()
    euclidean_distance, cosine_distance, cosine_sim_score, manhattan_distance = compute_distance(
        prompt, definition)
    print(euclidean_distance, cosine_distance,
          cosine_sim_score, manhattan_distance)
