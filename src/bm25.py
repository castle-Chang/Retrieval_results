import math
from collections import Counter, defaultdict


class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents
        self.N = len(documents)
        self.avgdl = sum(d["doc_len"] for d in documents) / (self.N or 1)
        self.k1 = k1
        self.b = b
        df = defaultdict(int)
        for d in documents:
            seen = set(d["tokens"])
            for t in seen:
                df[t] += 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def score(self, query_tokens, doc):
        f = Counter(doc["tokens"])
        dl = doc["doc_len"]
        s = 0.0
        for t in query_tokens:
            if t not in self.idf:
                continue
            idf = self.idf[t]
            tf = f[t]
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += idf * (tf * (self.k1 + 1)) / (denom or 1)
        return s

    def topk(self, query_tokens, k=10):
        scored = [(self.score(query_tokens, d), d) for d in self.documents]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]
