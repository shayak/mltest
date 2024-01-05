import time
import faiss
import requests
from io import StringIO
import pandas as pd

from sentence_transformers import SentenceTransformer


def prep_data():
    res = requests.get('https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/sick2014/SICK_train.txt')
    # create dataframe
    data = pd.read_csv(StringIO(res.text), sep='\t')
    data.head()

    sentences = data['sentence_A'].tolist()
    sentence_b = data['sentence_B'].tolist()
    sentences.extend(sentence_b)

    urls = [
        'https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2012/MSRpar.train.tsv',
        'https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2012/MSRpar.test.tsv',
        'https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2012/OnWN.test.tsv',
        'https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2013/OnWN.test.tsv',
        'https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2014/OnWN.test.tsv',
        'https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2014/images.test.tsv',
        'https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2015/images.test.tsv'
    ]

    # each of these dataset have the same structure, so we loop through each creating our sentences data
    for url in urls:
        res = requests.get(url)
        # extract to dataframe
        data2 = pd.read_csv(StringIO(res.text), sep='\t', header=None, on_bad_lines='skip')
        # add to columns 1 and 2 to sentences list
        sentences.extend(data2[1].tolist())
        sentences.extend(data2[2].tolist())

    sentences = [word for word in list(set(sentences)) if type(word) is str]

    return data, sentences


def build_index(model, data):
    sentence_embeddings = model.encode(data)
    d = sentence_embeddings.shape[1]  # cols


    # without quantization, search takes ~0.004...s
    # with quantization, search takes ~0.00011...s

    flat_index = faiss.IndexFlatL2(d)

    # this quantization assigns vectors to a centroid (voronoi cells), but does not chop of precision when storing
    # index = faiss.IndexIVFFlat(flat_index, d, 50)

    # this quantization chops of precision (less bits)
    m = 8  # number of centroids
    bits = 8
    index = faiss.IndexIVFPQ(flat_index, d, 50, m, bits)

    index.train(sentence_embeddings)
    index.add(sentence_embeddings)

    print(index.ntotal)
    return index


def search(model, index, k, query):  # k = num matches
    xq = model.encode([query])
    D, I = index.search(xq, k)


def run():
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    ogdata, data = prep_data()
    index = build_index(model, data)

    k = 4
    # xq = model.encode(["Someone sprints with a football"])
    xq = model.encode(["I want to fly away"])
    start = time.perf_counter()
    D, I = index.search(xq, k)
    end = time.perf_counter()
    print(I)
    print(f'Time taken: {end-start}s\n')
    results = [data[i] for i in I[0]]
    print(results)


if __name__ == '__main__':
    run()
