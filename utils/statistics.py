# import sqlite3
# from database.queries import get_all_definitions, insert_statistic, get_definition_by_id
import statistics

def count_words(definition):
    words = definition.split(' ')
    return len(words)


def unique_words(definition):
    unique = len(set(definition.split(' ')))
    return unique


def count_sentence(definition):
    sentence = definition.split('.')
    return len(sentence)


def count_paragraph(definition):
    paragraph = definition.split('\n\n')
    return len(paragraph)


def compute_statistic(definition):
    words = count_words(definition)
    unique = unique_words(definition)
    sentence = count_sentence(definition)
    paragraph = count_paragraph(definition)

    return unique, words, sentence, paragraph


def compute_mean_statistics(definitions):
    unq = []
    wrd = []
    snt = []
    prg = []

    for definition in definitions:
        unique, words, sentence, paragraph = compute_statistic(definition)
        unq.append(unique)
        wrd.append(words)
        snt.append(sentence)
        prg.append(paragraph)

    statistic = {
        'unq': round(statistics.mean(unq), 3),
        'wrd': round(statistics.mean(wrd), 3),
        'snt': round(statistics.mean(snt), 3),
        'prg': round(statistics.mean(prg), 3),
    }

    return statistic