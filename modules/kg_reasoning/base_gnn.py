import torch
import numpy as np
from collections import defaultdict

VERY_NEG_NUMBER = -100000000000

class BaseGNNLayer(torch.nn.Module):
    """
    Builds sparse tensors that represent structure.
    """
    def __init__(self, args, num_entity, num_relation):
        super(BaseGNNLayer, self).__init__()
        self.num_relation = num_relation
        self.num_entity = num_entity
        self.device = torch.device('cuda' if args['use_cuda'] else 'cpu')
        self.normalized_gnn = args['normalized_gnn']


    def build_matrix(self):
        batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list = self.edge_list
        num_fact = len(fact_ids)
        num_relation = self.num_relation
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        self.num_fact = num_fact
        fact2head = torch.LongTensor([batch_heads, fact_ids]).to(self.device)
        fact2tail = torch.LongTensor([batch_tails, fact_ids]).to(self.device)
        head2fact = torch.LongTensor([fact_ids, batch_heads]).to(self.device)
        tail2fact = torch.LongTensor([fact_ids, batch_tails]).to(self.device)
        rel2fact = torch.LongTensor([fact_ids, batch_rels + batch_ids * num_relation]).to(self.device)
        fact2rel = torch.LongTensor([batch_rels + batch_ids * num_relation, fact_ids]).to(self.device)
        self.batch_rels = torch.LongTensor(batch_rels).to(self.device)
        self.batch_ids = torch.LongTensor(batch_ids).to(self.device)
        self.batch_heads = torch.LongTensor(batch_heads).to(self.device)
        self.batch_tails = torch.LongTensor(batch_tails).to(self.device)
        # self.batch_ids = batch_ids
        if self.normalized_gnn:
            vals = torch.FloatTensor(weight_list).to(self.device)
        else:
            vals = torch.ones_like(self.batch_ids).float().to(self.device)

        #vals = torch.ones_like(self.batch_ids).float().to(self.device)
        # Sparse Matrix for reason on graph
        self.fact2head_mat = self._build_sparse_tensor(fact2head, vals, (batch_size * max_local_entity, num_fact))
        self.head2fact_mat = self._build_sparse_tensor(head2fact, vals, (num_fact, batch_size * max_local_entity))
        self.fact2tail_mat = self._build_sparse_tensor(fact2tail, vals, (batch_size * max_local_entity, num_fact))
        self.tail2fact_mat = self._build_sparse_tensor(tail2fact, vals, (num_fact, batch_size * max_local_entity))
        self.fact2rel_mat = self._build_sparse_tensor(fact2rel, vals, (batch_size * num_relation, num_fact))
        self.rel2fact_mat = self._build_sparse_tensor(rel2fact, vals, (num_fact, batch_size * num_relation))

        self.fact2entity_mat = self.head2fact_mat + self.tail2fact_mat
        relation_list = list(set(batch_rels))
        # rel2fact = np.zeros((len(relation_list),len(batch_rels)))
        index_dict = -np.ones(max(relation_list) + 1)
        index_dict[relation_list] = range(len(relation_list))
        fact_index = index_dict[batch_rels]
        rel2fact_dense = torch.from_numpy(np.array([fact_index, list(np.arange(len(batch_rels)))])).long().to(
            self.device)
        fact2rel_dense = torch.from_numpy(np.array([list(np.arange(len(batch_rels))), fact_index])).long().to(
            self.device)
        self.rel2fact_mat_dense = self._build_sparse_tensor(rel2fact_dense, vals, (len(relation_list), num_fact))
        self.fact2rel_mat_dense = self._build_sparse_tensor(fact2rel_dense, vals, (num_fact, len(relation_list)))
        # print("hi",self.rel2fact_mat_dense,self.fact2entity_mat)
        self.rel2entity_mat = torch.sparse.mm(self.rel2fact_mat_dense, self.fact2entity_mat)
        self.rel_adj = torch.sparse.mm(self.rel2entity_mat, self.rel2entity_mat.t())
        self.rel_adj = torch.where(self.rel_adj.to_dense() > 0, 1, 0)

    def _build_sparse_tensor(self, indices, values, size):
        return torch.sparse.FloatTensor(indices, values, size).to(self.device)
