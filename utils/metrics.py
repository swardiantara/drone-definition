from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, manhattan_distances
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer


def compute_distance(prompt, definition, model='instructor'):
    # Instantiate the Embedding Model (T5 Encoder or all-MiniLM-L6-v2)
    embedder = INSTRUCTOR('hkunlp/instructor-base') if model == 'instructor' else SentenceTransformer(
        'sentence-transformers/all-MiniLM-L6-v2')
    query = [
        ['Represent the Wikipedia question for retrieving supporting documents: ', prompt]]
    document = [['Represent the Wikipedia document for retrieval: ', definition]]

    # Compute the embedding vector
    query_embedding = embedder.encode(query).reshape(
        1, -1) if model == 'instructor' else embedder.encode(prompt).reshape(1, -1)
    document_embedding = embedder.encode(document).reshape(
        1, -1) if model == 'instructor' else embedder.encode(definition).reshape(1, -1)

    # Compute the distance metrics
    [[euclidean_distance]] = euclidean_distances(
        query_embedding, document_embedding)
    [[cosine_distance]] = cosine_distances(query_embedding, document_embedding)
    [[manhattan_distance]] = manhattan_distances(
        query_embedding, document_embedding)
    return euclidean_distance, cosine_distance, manhattan_distance
