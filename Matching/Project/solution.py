from typing import List, Dict
from flask import Flask, jsonify, json, request
from langdetect import detect
import os
import numpy as np
import re
import nltk
import string
import faiss
import torch

app = Flask(__name__)

emb_path_knrm = os.getenv('EMB_PATH_KNRM') if os.getenv('EMB_PATH_KNRM') else 'D:/Git Projects/Matching/Project/data/embedings.bin'
vocab_path = os.getenv('VOCAB_PATH') if os.getenv('VOCAB_PATH') else 'D:/Git Projects/Matching/Project/data/vocab.json'
mlp_path = os.getenv('MLP_PATH') if os.getenv('MLP_PATH') else 'D:/Git Projects/Matching/Project/data/mlp.bin'
emb_path_glove = os.getenv('EMB_PATH_GLOVE') if os.getenv('EMB_PATH_GLOVE') else 'D:/Git Projects/Matching/Project/data/glove.6B.50d.txt'


class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        return torch.exp(
            -torch.pow(x - self.mu, 2) / (2 * self.sigma**2)
        )


class KNRM(torch.nn.Module):
    def __init__(
            self,
            # num_emb: int,
            # emb_dim: int,
            embedding_state_dict: Dict[str, any],
            mlp_state_dict: Dict[str, any]
        ):
        super().__init__()

        self.emb_shape = embedding_state_dict['weight'].shape
        self.embeddings = torch.nn.Embedding.from_pretrained(
            embedding_state_dict['weight'],
            freeze=True,
            padding_idx=0
        )
        # self.embeddings = torch.nn.Embedding(
        #     num_embeddings=num_emb,
        #     embedding_dim=emb_dim,
        #     padding_idx=0
        # )
        # self.embeddings.load_state_dict(embedding_state_dict)

        self.kernel_num = 21
        self.sigma = 0.1
        self.exact_sigma = 0.001
        self.kernels = self._get_kernels_layers()

        self.out_layers = [10, 5]
        self.mlp = self._get_mlp()
        self.mlp.load_state_dict(mlp_state_dict)

        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        kernels = torch.nn.ModuleList()
        threshold = 2 / (self.kernel_num - 1)
        global_m = -1
        for k in range(self.kernel_num):
            if k == self.kernel_num - 1:
                kernels.append(GaussianKernel(mu= 1, sigma=self.exact_sigma))
            else:
                global_m += threshold / 2 if k == 0 else threshold
                kernels.append(GaussianKernel(mu=global_m, sigma=self.sigma))
        return kernels

    def _get_mlp(self) -> torch.nn.Sequential:
        module_list = []
        prev_neurons = len(self.kernels)
        for rel in self.out_layers:
            module_list.append(torch.nn.ReLU())
            module_list.append(torch.nn.Linear(prev_neurons, rel))
            prev_neurons = rel
        if len(self.out_layers) != 0:
            module_list.append(torch.nn.ReLU())
        module_list.append(torch.nn.Linear(prev_neurons, 1))
        
        seq = torch.nn.Sequential(*module_list)
        return seq

    def forward(self, input_1: Dict[str, torch.Tensor], input_2: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)
        return out

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        query = self.embeddings(query)
        doc = self.embeddings(doc)
        query_norm = query / (query.norm(p=2, dim=-1, keepdim=True) + 1e-13)
        doc_norm = doc / (doc.norm(p=2, dim=-1, keepdim=True) + 1e-13)
        return torch.bmm(query_norm, doc_norm.transpose(-1, -2))

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        query, doc = inputs['query'], inputs['document']

        matching_matrix = self._get_matching_matrix(query, doc)
        kernels_out = self._apply_kernels(matching_matrix)
        out = self.mlp(kernels_out)

        return out


class Solution():
    def __init__(self) -> None:
        self.knrm = None
        self.vocab = None
        self.emb_glove = None
        self.index = None
        
        self.docs = {}
        self.is_initialized = False

        self._initialize()

    def _initialize(self):
        def _read_glove_embeddings(file_path: str) -> Dict[str, List[str]]:
            embeddings_dict = {}
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], "float32")
                    embeddings_dict[word] = vector
            return embeddings_dict
        
        emb_dict_state = torch.load(emb_path_knrm)
        vocab = json.load(open(vocab_path))
        mlp_dict_state = torch.load(mlp_path)
        emb_glove = _read_glove_embeddings(emb_path_glove)

        knrm = KNRM(
            # num_emb=len(vocab),
            # emb_dim=50,
            embedding_state_dict=emb_dict_state, 
            mlp_state_dict=mlp_dict_state
        )

        self.knrm = knrm
        self.vocab = vocab
        self.emb_glove = emb_glove
        self.is_initialized = True

    def _hadle_punctuation(self, inp_str: str) -> str:
        inp_str = inp_str.replace("\\", "/")
        return re.sub(f"[{string.punctuation}]", " ", inp_str)

    def _simple_preproc(self, inp_str: str) -> List[str]:
        res = nltk.word_tokenize(self._hadle_punctuation(inp_str.lower()))
        return res

    def _tokenize(self, text: str) -> torch.LongTensor:
        tokens = self._simple_preproc(text)
        return torch.LongTensor([self.vocab.get(x, self.vocab['OOV']) for x in tokens])

    def _vectorize(self, text: str) -> np.ndarray:
        tokens_ids = self._tokenize(text)
        embendigs = self.knrm.embeddings(tokens_ids)
        return embendigs.mean(axis=0).numpy() 
        
    def update_index(self, vectors: np.ndarray, ids: np.ndarray):
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index = faiss.IndexIDMap(self.index)
        self.index.add_with_ids(
            vectors,
            ids 
        )

    def predict(self, query: str, ids: np.ndarray) -> np.ndarray:
        queries = []
        docs = []

        query = self._tokenize(query).tolist()
        max_len_d1 = -1
        for i in ids:
            document = self._tokenize(self.docs[str(i)]).tolist()
            max_len_d1 = max(len(document), max_len_d1)
        for i in ids:
            document = self._tokenize(self.docs[str(i)]).tolist()
            queries.append(query)
            docs.append(document + [0] * (max_len_d1 - len(document)))
        
        texts = {
            'query': torch.LongTensor(queries),
            'document': torch.LongTensor(docs)
        }

        preds = self.knrm.predict(texts).flatten()

        sorted_id_texts = torch.sort(preds, descending=True).indices.numpy()
        ids = ids[sorted_id_texts]
        return ids

sol = Solution()


@app.route('/ping')
def ping():
    res =  {'status': 'work'}
    if sol.is_initialized:
        res['status'] = 'ok'
    return res
    
@app.route('/query', methods=['POST'])
def query():
    if (sol.index is None or sol.index.ntotal == 0):
        return jsonify({
            'status': 'FAISS is not initialized!'
        })
    
    # Prod
    queries = json.loads(request.json)['queries']
    # Test
    # queries = json.loads(json.dumps(request.json))['queries']
    res_lungs = []
    res_suggestions = []

    for current_query in queries:
        if (detect(current_query) != 'en'):
            res_lungs.append(False)
            res_suggestions.append(None)
            continue

        res_lungs.append(True)
        current_vector = sol._vectorize(current_query).reshape(-1, sol.knrm.emb_shape[1])
        _, ids = sol.index.search(current_vector, 10)

        ids = ids.reshape(-1)
        ids = sol.predict(current_query, ids)

        res_suggestions.append([
            (str(i), sol.docs[str(i)]) for i in ids
        ])
    return {
        'lang_check': res_lungs,
        'suggestions': res_suggestions
    }

    
@app.route('/update_index', methods=['POST'])
def update_index():
    # Prod
    docs = json.loads(request.json)['documents']
    # Test
    # docs = json.loads(json.dumps(request.json))['documents']
    doc_vectors = []
    doc_indexs = []

    for id, text in docs.items():
        doc_vectors.append(sol._vectorize(text=text))
        doc_indexs.append(int(id))
        sol.docs.update({id: text})

    vectors = np.array(doc_vectors)
    ids = np.array(doc_indexs)
    
    sol.update_index(vectors=vectors, ids=ids)

    return {
        'status': 'ok',
        'index_size': sol.index.ntotal
    }
