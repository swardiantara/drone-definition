import os
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, cosine_distances, manhattan_distances
from InstructorEmbedding import INSTRUCTOR


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
        query_embedding = model.encode(query)
        document_embedding = model.encode(definition)
        euclidean_distance = euclidean_distances(
            query_embedding, document_embedding)
        cosine_distance = cosine_distances(query_embedding, document_embedding)
        manhattan_distance = manhattan_distances(
            query_embedding, document_embedding)
        with open('{}_{}.txt'.format('chatGP', term), 'w') as file:
            file.write(definition + '\n' + 'Euclidean Distance: {}'.format(euclidean_distance) + '\n' +
                       'Cosine Distance: {}'.format(cosine_distance) + '\n' + 'Manhattan Distance: {}'.format(manhattan_distance))
        print(term)
        print(entity_type)
        print(definition)


if __name__ == "__main__":
    main()
