from util import *


class HGTLayer(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 num_heads: int,
                 node_types: set[str],
                 edge_types: set[str],
                 dropout_ratio: float = 0.2,
                 use_norm: bool = False):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.node_types = node_types
        self.edge_types = edge_types
        self.num_node_types = len(node_types)
        self.num_edge_types = len(edge_types)
        self.use_norm = use_norm
        self.head_dim = out_dim // num_heads
        self.sqrt_head_dim = math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(dropout_ratio)

        self.k_linears = nn.ModuleDict()
        self.q_linears = nn.ModuleDict()
        self.v_linears = nn.ModuleDict()
        self.a_linears = nn.ModuleDict()
        self.norms = nn.ModuleDict()

        for node_type in node_types:
            self.k_linears[node_type] = nn.Linear(in_dim, out_dim)
            self.q_linears[node_type] = nn.Linear(in_dim, out_dim)
            self.v_linears[node_type] = nn.Linear(in_dim, out_dim)
            self.a_linears[node_type] = nn.Linear(out_dim, out_dim)

            if use_norm:
                self.norms[node_type] = nn.LayerNorm(out_dim)
                
        self.relation_pri = nn.ParameterDict({
            edge_type: nn.Parameter(torch.ones(self.num_heads))
            for edge_type in self.edge_types
        })
        self.relation_att = nn.ParameterDict({
            edge_type: nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.head_dim))
            for edge_type in self.edge_types
        })
        self.relation_msg = nn.ParameterDict({
            edge_type: nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.head_dim))
            for edge_type in self.edge_types
        })

        self.skip = nn.ParameterDict({
            node_type: nn.Parameter(torch.tensor(1.))
            for node_type in self.node_types
        })

        for parameter in itertools.chain(self.relation_att.values(), self.relation_msg.values()):
            nn.init.xavier_uniform_(parameter)
        
    def forward(self,
                hg: dgl.DGLHeteroGraph,
                node_feat: dict[str, FloatTensor]) -> dict[str, FloatTensor]:
        with hg.local_scope():
            for src_type, edge_type, dest_type in hg.canonical_etypes:
                subgraph = hg[src_type, edge_type, dest_type]
                # assert subgraph.is_homogeneous
                
                k_linear = self.k_linears[src_type]
                v_linear = self.v_linears[src_type]
                q_linear = self.q_linears[dest_type]

                k = k_linear(node_feat[src_type])
                v = v_linear(node_feat[src_type])
                q = q_linear(node_feat[dest_type])
                
                k = k.view(-1, self.num_heads, self.head_dim)
                v = v.view(-1, self.num_heads, self.head_dim)
                q = q.view(-1, self.num_heads, self.head_dim)
                
                relation_att = self.relation_att[edge_type]
                relation_pri = self.relation_pri[edge_type]
                relation_msg = self.relation_msg[edge_type]
                
                # [num_nodes, num_heads, head_dim], [num_heads, head_dim, head_dim] -> [num_nodes, num_heads, head_dim]
                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                subgraph.srcdata['k'] = k 
                subgraph.dstdata['q'] = q 
                subgraph.srcdata[f'v_{edge_type}'] = v 
                
                # sub_graph.edata['t']: [num_edges, num_heads, 1]
                subgraph.apply_edges(
                    dglfn.u_dot_v('k', 'q', 't')
                )
                
                # [num_edges, num_heads]
                attn_score = torch.sum(subgraph.edata.pop('t'), dim=-1) \
                             * relation_pri / self.sqrt_head_dim
                attn_score = dglF.edge_softmax(graph=subgraph,
                                               logits=attn_score,
                                               norm_by='dst')
                
                # sub_graph.edata['t']: [num_edges, num_heads, 1]
                subgraph.edata['t'] = torch.unsqueeze(attn_score, dim=-1)
                
            hg.multi_update_all(etype_dict={
                                    edge_type: (dglfn.u_mul_e(f'v_{edge_type}', 't', 'm'), dglfn.sum('m', 't'))
                                    for edge_type in self.edge_types
                                },
                                cross_reducer='mean')
            
            new_node_feat = {}
            
            for node_type in hg.ntypes:
                # float-scalar
                alpha = torch.sigmoid(self.skip[node_type])
                
                # [num_nodes, num_heads, head_dim]
                t = hg.nodes[node_type].data['t']
                
                # [num_nodes, out_dim]
                t = t.view(-1, self.out_dim)
                
                # [num_nodes, out_dim]
                trans_out = self.dropout(
                    self.a_linears[node_type](t)
                )
                
                trans_out = trans_out * alpha + node_feat[node_type] * (1 - alpha)
                
                if self.use_norm:
                    new_node_feat[node_type] = self.norms[node_type](trans_out)
                else:
                    new_node_feat[node_type] = trans_out
                
            return new_node_feat
                
                
class HGT(nn.Module):
    def __init__(self,
                 hg: dgl.DGLHeteroGraph,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 num_layers: int,
                 num_heads: int,
                 use_norm: bool = True):
        super().__init__()
        
        self.node_types = set(hg.ntypes)
        self.edge_types = set(hg.etypes)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        self.hgt_layers = nn.ModuleList()
        self.adapt_ws = nn.ModuleDict() 
        
        for node_type in self.node_types:
            self.adapt_ws[node_type] = nn.Linear(in_dim, hidden_dim)
            
        for _ in range(num_layers):
            self.hgt_layers.append(
                HGTLayer(in_dim=hidden_dim,
                         out_dim=hidden_dim,
                         num_heads=num_heads,
                         node_types=self.node_types,
                         edge_types=self.edge_types,
                         use_norm=use_norm)
            )
            
        self.out_linear = nn.Linear(hidden_dim, out_dim)

    def forward(self,
                hg: dgl.DGLHeteroGraph,
                out_node_type: str) -> FloatTensor:
        node_feat = {}
        
        for node_type in hg.ntypes:
            node_feat[node_type] = F.gelu(
                self.adapt_ws[node_type](
                    hg.nodes[node_type].data['inp']
                )
            )
            
        for hgt_layer in self.hgt_layers:
            node_feat = hgt_layer(hg, node_feat)
            
        out = self.out_linear(
            node_feat[out_node_type]
        )
        
        return out 
