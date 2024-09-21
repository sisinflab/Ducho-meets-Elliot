"""
Module description:

"""

from abc import ABC

from .NGCFLayer import NGCFLayer
import torch
import torch_geometric
import numpy as np
import random

from torch_sparse import SparseTensor


class NGCFMModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 l_w,
                 weight_size,
                 n_layers,
                 multimodal_features,
                 message_dropout,
                 random_seed,
                 name="NGCFM",
                 **kwargs
                 ):
        super().__init__()

        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.weight_size = weight_size
        self.n_layers = n_layers
        self.message_dropout = message_dropout
        self.weight_size_list = [self.embed_k] + ([self.weight_size] * self.n_layers)

        self.Gu = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(self.num_users, self.embed_k)))
        self.Gu.to(self.device)
        self.Gi = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(self.num_items, self.embed_k)))
        self.Gi.to(self.device)

        # multimodal
        self.Tu = torch.nn.Embedding(self.num_users, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Tu.weight)
        self.Tu.to(self.device)
        self.F = torch.tensor(multimodal_features, dtype=torch.float32, device=self.device)
        self.F.to(self.device)
        self.feature_shape = multimodal_features.shape[1]
        self.proj = torch.nn.Linear(in_features=self.feature_shape, out_features=self.embed_k)
        self.proj.to(self.device)

        propagation_network_list = []
        self.dropout_layers = []

        for layer in range(self.n_layers):
            propagation_network_list.append((NGCFLayer(self.weight_size_list[layer],
                                                       self.weight_size_list[layer + 1]), 'x, edge_index -> x'))
            self.dropout_layers.append(torch.nn.Dropout(p=self.message_dropout))

        self.propagation_network = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)
        self.propagation_network.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def propagate_embeddings(self, adj):
        ego_embeddings = torch.cat((self.Gu.to(self.device), self.Gi.to(self.device)), 0)
        all_embeddings = [ego_embeddings]
        embedding_idx = 0

        for layer in range(self.n_layers):
            all_embeddings += [torch.nn.functional.normalize(self.dropout_layers[embedding_idx](list(
                self.propagation_network.children()
            )[layer](all_embeddings[embedding_idx].to(self.device), adj.to(self.device))), p=2, dim=1)]
            embedding_idx += 1

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        gu, gi = torch.split(all_embeddings, [self.num_users, self.num_items], 0)
        return gu, gi

    def forward(self, inputs, **kwargs):
        gu, gi, users, items = inputs
        gamma_u = torch.squeeze(gu).to(self.device)
        gamma_i = torch.squeeze(gi).to(self.device)
        theta_u = torch.squeeze(self.Tu.weight[users]).to(self.device)
        effe_i = torch.squeeze(self.F[items]).to(self.device)
        proj_i = torch.nn.functional.normalize(self.proj(effe_i).to(self.device), p=2, dim=1)

        xui = torch.sum(gamma_u * gamma_i, 1) + torch.sum(theta_u * proj_i, 1)

        return xui, gamma_u, gamma_i, theta_u, proj_i

    def predict(self, gu, gi, start_user, stop_user, **kwargs):
        P = torch.nn.functional.normalize(self.proj(self.F).to(self.device), p=2, dim=1)
        return torch.matmul(gu.to(self.device), torch.transpose(gi.to(self.device), 0, 1)) + \
               torch.matmul(self.Tu.weight[start_user:stop_user].to(self.device), torch.transpose(P.to(self.device), 0, 1))

    def train_step(self, batch, adj):
        gu, gi = self.propagate_embeddings(adj)
        user, pos, neg = batch
        xu_pos, gamma_u, gamma_i_pos, theta_u, proj_i_pos = self.forward(inputs=(gu[user], gi[pos], user, pos))
        xu_neg, _, gamma_i_neg, _, proj_i_neg = self.forward(inputs=(gu[user], gi[neg], user, neg))

        difference = torch.clamp(xu_pos - xu_neg, -80.0, 1e8)
        loss = torch.mean(torch.nn.functional.softplus(-difference))
        reg_loss = self.l_w * (1 / 2) * (gamma_u.norm(2).pow(2) +
                                         gamma_i_pos.norm(2).pow(2) +
                                         gamma_i_neg.norm(2).pow(2) +
                                         theta_u.norm(2).pow(2) +
                                         proj_i_pos.norm(2).pow(2) +
                                         proj_i_neg.norm(2).pow(2)) / len(user)
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
