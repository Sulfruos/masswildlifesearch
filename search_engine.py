import torch
import string
import json
import os
import faiss
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
from rank_bm25 import BM25Okapi

class WildlifeSearchEngine:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu" # trying mps as I'm using an M3
        print("Using device: " + self.device)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.corpus = self._load_corpus()
        self.faiss_index = self._build_faiss_index()
        self.bm25_index = self._build_bm25_index()

        # self.qa_model = AutoModelForCausalLM.from_pretrained(
        # "microsoft/phi-2",
        # torch_dtype=torch.float32,
        # trust_remote_code=True
        # ).to("cpu")
        # self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        self.qa_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        torch_dtype=torch.float16, 
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
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
            section = " ".join((doc["section"]).split('_'))
            enhanced_text = f"{section} {doc["content"]} {categories_text}"
            print(enhanced_text)
            enhanced_corpus.append(enhanced_text)
        embeddings = self.model.encode(enhanced_corpus, convert_to_tensor=False)
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
            text = text.replace('-', ' ')
            text = text.translate(str.maketrans('', '', string.punctuation))
            return text.split()
        
        # Use the same enhanced corpus as FAISS
        enhanced_corpus = []
        for doc in self.corpus:
            categories_text = " ".join(doc.get("categories", []))
            section = " ".join((doc["section"]).split('_'))
            enhanced_text = f"{section} {doc["content"]} {categories_text}"
            # print(enhanced_text)
            enhanced_corpus.append(enhanced_text)
        
        tokenized_corpus = [preprocess_text(doc) for doc in enhanced_corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        return bm25

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
            print(f"Source: {scores[idx]:.4f}", self.corpus[idx]["source"])
            results.append({"score": scores[idx], "content": self.corpus[idx]["content"], "source": self.corpus[idx]["source"]})
            

        return results
    
    def generate_rag_answer(self, query):

        # get top k relevant docs for rag and get their context chunks
        results = self.query_docs(query)
        context_chunks = [res["content"] for res in results]

        context = "\n\n".join([f"Context {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])
        
    #     prompt = f"""Based on the provided context, answer the following question concisely and accurately.

    # {context}

    # Question: {query}
    # Answer:"""

        prompt = f"""<s>[INST] Based on the provided context about Massachusetts wildlife, answer the following question concisely and accurately. Provide a complete answer in 2-3 sentences.

        Context:
        {context}

        Question: {query} [/INST]"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()} # move inputs to same device as model
        
        with torch.no_grad():
            outputs = self.qa_model.generate(
                **inputs,
                max_new_tokens=200,  
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract only the new tokens (the answer)
        input_length = inputs['input_ids'].shape[1]
        answer = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        # Clean up common issues
        answer = answer.split("Question:")[0].strip()  # Stop at next question
        answer = answer.split("\n\n")[0].strip()      # Stop at paragraph break
        
        return results, answer

if __name__ == "__main__":
    my_search_engine = WildlifeSearchEngine()

    query = "What should I do if I see a bear?"
    # results = my_search_engine.query_docs(query)

    # chunks = [res["content"] for res in results]
    # answer = my_search_engine.generate_rag_answer(chunks, query)
    # print(answer)

    results, answer = my_search_engine.generate_rag_answer(query)

    print(answer)