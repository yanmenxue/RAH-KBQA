
import torch
import torch.nn.functional as F
import torch.nn as nn


from .base_gnn import BaseGNNLayer

VERY_NEG_NUMBER = -100000000000

class ReasonGNNLayer(BaseGNNLayer):
    """
    GNN Reasoning
    """
    def __init__(self, args, num_entity, num_relation, entity_dim, alg):
        super(ReasonGNNLayer, self).__init__(args, num_entity, num_relation)
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.entity_dim = entity_dim
        self.alg = alg
        self.num_ins = args['num_ins']
        self.num_gnn = args['num_gnn']
        
        self.init_layers(args)

    def init_layers(self, args):
        entity_dim = self.entity_dim
        self.softmax_d1 = nn.Softmax(dim=1)
        self.score_func = nn.Linear(in_features=entity_dim, out_features=1)
        self.glob_lin = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        self.lin = nn.Linear(in_features=2*entity_dim, out_features=entity_dim)
        assert self.alg == 'bfs'
        self.linear_dropout = args['linear_dropout']
        self.linear_drop = nn.Dropout(p=self.linear_dropout)
        for i in range(self.num_gnn):
            self.add_module('rel_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            if self.alg == 'bfs':
                self.add_module('e2e_linear' + str(i), nn.Linear(in_features=2*(self.num_ins)*entity_dim + entity_dim, out_features=entity_dim))
        self.lin_m =  nn.Linear(in_features=(self.num_ins)*entity_dim, out_features=entity_dim)

        self.add_module('e2r_linear' , nn.Linear(in_features=2 * entity_dim, out_features=entity_dim))
        self.add_module('r2r_linear', nn.Linear(in_features=2 * entity_dim, out_features=entity_dim))
        self.add_module('r2r_attention_linear', nn.Linear(in_features=entity_dim, out_features=entity_dim))
        self.add_module('r2e_linear' , nn.Linear(in_features=3 * entity_dim, out_features=entity_dim))

    def init_reason(self, local_entity, kb_adj_mat, local_entity_emb, rel_features, rel_features_inv, query_entities, query_node_emb=None):
        batch_size, max_local_entity = local_entity.size()
        self.local_entity_mask = (local_entity != self.num_entity).float()
        self.batch_size = batch_size
        self.max_local_entity = max_local_entity
        self.edge_list = kb_adj_mat
        self.rel_features = rel_features
        self.rel_features_inv = rel_features_inv
        self.local_entity_emb = local_entity_emb
        self.num_relation = self.rel_features.size(0)
        self.possible_cand = []
        self.build_matrix()
        self.query_entities = query_entities

        self.fact_rel = torch.index_select(self.rel_features, dim=0, index=self.batch_rels)
       

    def reason_layer(self, curr_dist, instruction, rel_linear):
        """
        Aggregates neighbor representations
        """
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        # num_relation = self.num_relation
        rel_features = self.rel_features
        
        fact_rel = torch.index_select(rel_features, dim=0, index=self.batch_rels)
        
        fact_query = torch.index_select(instruction, dim=0, index=self.batch_ids)
        fact_val = F.relu(rel_linear(fact_rel) * fact_query)
        fact_prior = torch.sparse.mm(self.head2fact_mat, curr_dist.view(-1, 1))

        fact_val = fact_val * fact_prior
        
        f2e_emb = torch.sparse.mm(self.fact2tail_mat, fact_val)
        assert not torch.isnan(f2e_emb).any()

        neighbor_rep = f2e_emb.view(batch_size, max_local_entity, self.entity_dim)
        
        return neighbor_rep

    def reason_layer_inv(self, curr_dist, instruction, rel_linear):
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        # num_relation = self.num_relation
        rel_features = self.rel_features_inv
        
        fact_rel = torch.index_select(rel_features, dim=0, index=self.batch_rels)
        
        fact_query = torch.index_select(instruction, dim=0, index=self.batch_ids)
        fact_val = F.relu(rel_linear(fact_rel) * fact_query)
        fact_prior = torch.sparse.mm(self.tail2fact_mat, curr_dist.view(-1, 1))
        

        fact_val = fact_val * fact_prior

        f2e_emb = torch.sparse.mm(self.fact2head_mat, fact_val)
        assert not torch.isnan(f2e_emb).any()

        neighbor_rep = f2e_emb.view(batch_size, max_local_entity, self.entity_dim)
        
        return neighbor_rep

    def combine(self,emb):
        """
        Combines instruction-specific representations.
        """
        local_emb = torch.cat(emb, dim=-1)
        local_emb = F.relu(self.lin_m(local_emb))

        score_func = self.score_func
        
        score_tp = score_func(self.linear_drop(local_emb)).squeeze(dim=2)
        answer_mask = self.local_entity_mask
        self.possible_cand.append(answer_mask)
        score_tp = score_tp + (1 - answer_mask) * VERY_NEG_NUMBER
        current_dist = self.softmax_d1(score_tp)
        return current_dist, local_emb

    def forward(self, current_dist, relational_ins, step=0, return_score=False):
        """
        Compute next probabilistic vectors and current node representations.
        """
        rel_linear = getattr(self, 'rel_linear' + str(step))
        e2e_linear = getattr(self, 'e2e_linear' + str(step))
        # score_func = getattr(self, 'score_func' + str(step))

        e2r_linear = getattr(self, 'e2r_linear' )
        r2r_linear = getattr(self, 'r2r_linear' )
        r2r_attention_linear = getattr(self, 'r2r_attention_linear' )
        r2e_linear = getattr(self, 'r2e_linear' )

        score_func = self.score_func
        neighbor_reps = []
        
        for j in range(relational_ins.size(1)):
            # we do the same procedure for existing and inverse relations
            neighbor_rep = self.reason_layer(current_dist, relational_ins[:,j,:], rel_linear)
            neighbor_reps.append(neighbor_rep)

            neighbor_rep = self.reason_layer_inv(current_dist, relational_ins[:,j,:], rel_linear)
            neighbor_reps.append(neighbor_rep)

        neighbor_reps = torch.cat(neighbor_reps, dim=2)
        
        
        next_local_entity_emb = torch.cat((self.local_entity_emb, neighbor_reps), dim=2)
        #print(next_local_entity_emb.size())
        self.local_entity_emb = F.relu(e2e_linear(self.linear_drop(next_local_entity_emb)))

        rel_features_unique = torch.sparse.mm(self.rel2fact_mat_dense, self.fact_rel)
        rel_features_unique = rel_features_unique / torch.sparse.sum(self.rel2fact_mat_dense,
                                                                     dim=1).to_dense().unsqueeze(1)
        n_rel = rel_features_unique.size()[0]
        rel_atte = r2r_attention_linear(rel_features_unique)
        attention = torch.mm(rel_atte, rel_features_unique.t())
        # a_input = torch.cat([rel_features_unique.repeat(1, n_rel).view(n_rel * n_rel, -1), \
        #                      rel_features_unique.repeat(n_rel, 1)], dim=1).view(n_rel, -1, 2 * self.entity_dim)
        # attention = F.relu(r2r_attention_linear(a_input)).squeeze(2)
        zero_vec = -1e12 * torch.ones_like(attention)
        attention = torch.where(self.rel_adj > 0, attention, zero_vec)
        attention = F.softmax(attention, dim=1)
        neighbor_rel_emb = torch.mm(attention, rel_features_unique)
        next_rel_emb = torch.cat((rel_features_unique, neighbor_rel_emb), dim=-1)
        next_rel_emb = F.leaky_relu(r2r_linear(next_rel_emb))
        self.fact_rel = torch.sparse.mm(self.fact2rel_mat_dense, next_rel_emb)

        next_fact_rel = torch.sparse.mm(self.head2fact_mat,
                                        self.local_entity_emb.view(-1, self.entity_dim)) - torch.sparse.mm(
            self.tail2fact_mat,
            self.local_entity_emb.view(-1, self.entity_dim))
        self.fact_rel = F.leaky_relu(e2r_linear(torch.cat((self.fact_rel, next_fact_rel), dim=-1)))

        btsize = self.local_entity_emb.size()[0]
        n_entity = self.local_entity_emb.size()[1]
        next_entity_head = torch.sparse.mm(self.fact2head_mat, self.fact_rel)
        num_head = torch.sparse.sum(self.fact2head_mat, dim=1).to_dense().unsqueeze(1).long()
        num_head = torch.where(num_head > 0, num_head, 1)
        next_entity_head /= num_head
        next_entity_head = next_entity_head.view(btsize, n_entity, self.entity_dim)
        next_entity_tail = torch.sparse.mm(self.fact2tail_mat, self.fact_rel)
        num_tail = torch.sparse.sum(self.fact2tail_mat, dim=1).to_dense().unsqueeze(1).long()
        num_tail = torch.where(num_tail > 0, num_tail, 1)
        next_entity_tail /= num_tail
        next_entity_tail = next_entity_tail.view(btsize, n_entity, self.entity_dim)
        self.local_entity_emb = F.leaky_relu(r2e_linear(torch.cat((self.local_entity_emb,
                                                                   next_entity_head, next_entity_tail),
                                                                  dim=-1)))

        score_tp = score_func(self.linear_drop(self.local_entity_emb)).squeeze(dim=2)
        answer_mask = self.local_entity_mask
        self.possible_cand.append(answer_mask)
        score_tp = score_tp + (1 - answer_mask) * VERY_NEG_NUMBER
        current_dist = self.softmax_d1(score_tp)
        if return_score:
            return score_tp, current_dist
        
        
        return current_dist, self.local_entity_emb 


