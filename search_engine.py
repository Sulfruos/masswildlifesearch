import torch
import string
import json
import os
import faiss
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

class WildlifeSearchEngine:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.corpus = self._load_corpus()
        self.faiss_index = self._build_faiss_index()
        self.bm25_index = self._build_bm25_index()
    
    def _load_corpus(self):
        corpus = []

        path_to_json_files = 'docs/'
        json_file_names = [filename for filename in os.listdir(path_to_json_files) if filename.endswith('.json')]

        for json_file_name in json_file_names:
            with open(os.path.join(path_to_json_files, json_file_name)) as json_file:
                data = json.load(json_file)
                for item in data:
                    corpus.append(item)

        return corpus
    
    def _create_build_faiss_index(self):
        enhanced_corpus = []
        for doc in self.corpus:
            categories_text = " ".join(doc.get("categories", []))
            print(categories_text)
            enhanced_corpus.append(f"{doc["content"]} {categories_text}")
        embeddings = self.model.encode_document(enhanced_corpus, convert_to_tensor=False)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, "nature_guide.index")
        print("Finished!")
        return index

    def _build_faiss_index(self):
        try:
            index = faiss.read_index("nature_guide.index")
            print("Loaded existing FAISS index")
            return index
        except:
            print("No existing index found, building new one...")
            return self._create_build_faiss_index()

    def _build_bm25_index(self):
        def preprocess_text(text):
            text = text.lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            return text.split()
        
        # Use the same enhanced corpus as FAISS
        enhanced_corpus = []
        for doc in self.corpus:
            categories_text = " ".join(doc.get("categories", []))
            enhanced_corpus.append(f"{doc['content']} {categories_text}")
        
        tokenized_corpus = [preprocess_text(doc) for doc in enhanced_corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        return bm25

    def update_faiss(self):
        ### update the nature guide index in faiss
        corpus_embeddings = self.model.encode_document(self.corpus, convert_to_tensor=False)
        index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
        index.add(corpus_embeddings)
        faiss.write_index(index, "nature_guide.index")

    def query_docs(self, query, top_k=3):

        ### evaluate queries and return top_k results for each

        # get scores from bm25 search
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        top_k_indices = sorted(range(len(bm25_scores)), key=lambda i : bm25_scores[i], reverse=True)[:top_k]

        # get hits from faiss search
        query_embedding_np = self.model.encode([query], convert_to_tensor=False)
        faiss_scores, faiss_indices = self.faiss_index.search(query_embedding_np, top_k)

        ### implementing reciprocal rank fusion to combine results (just doing BM25 + FAISS)
        scores = {}

        for d in range(len(self.corpus)):
            scores[d] = 0
        
        for rank, curr_d in enumerate(top_k_indices):
            scores[curr_d] += 1 / (10 + (rank + 1))

        for rank, idx in enumerate(faiss_indices[0]):
            scores[idx] += 2 / (10 + (rank + 1)) # double weight for FAISS 

        # after getting rrf scores for each doc, get the top k 
        
        rrf_top_k = sorted(scores, key=scores.get, reverse=True)[:top_k]

        results = []

        print("\nQuery:", query)

        # print("Top 3 FAISS matches in corpus:") # potentially outperforms RRF on smaller corpus
        # for score, idx in zip(faiss_scores[0], faiss_indices[0]):
        #     # print("------")
        #     # print(f"Score: {score:.4f}", corpus[idx])
        #     results.append({"score": float(score), "content": self.corpus[idx]["content"]})
        
        print("Top 3 RRF matches in corpus:")
        for idx in rrf_top_k:
            print("------")
            print(f"Score: {scores[idx]:.4f}", self.corpus[idx]["content"])
            results.append({"score": scores[idx], "content": self.corpus[idx]["content"]})

        return results

if __name__ == "__main__":

    my_search_engine = WildlifeSearchEngine()
    # my_search_engine.update_faiss()

    # Sample queries for running without using flask:
    # queries = [
    #     "How do black bears look?",
    #     "When do black bears mate?",
    #     "What is the black bear population in Massachusetts?",
    # ]

    # for query in queries:
    #     results = my_search_engine.query_docs(query)
    #     print(results)