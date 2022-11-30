import torch

import torch
import torch.nn as nn

def l2_normalize(x, axis=None, epsilon=1e-12):
  """l2 normalizes a tensor on an axis with numerical stability."""
  square_sum = torch.sum(torch.square(x), axis=axis, keepdims=True)
  x_inv_norm = torch.rsqrt(torch.maximum(square_sum, epsilon))
  return x * x_inv_norm

class PromptModule(nn.Module):
    def __init__(self, 
                prompt_len=5, 
                pool_size=50,
                topk=5,
                embed_dim=512, 
                prompt_key_init="uniform"):
        super().__init__()
        self.embed_dim = embed_dim
        self.prompt_len = prompt_len
        self.pool_size = pool_size
        self.topk = topk
        self.prompt = nn.parameter.Parameter(data=torch.empty(size=(self.pool_size, self.prompt_len, self.embed_dim)))
        self.prompt_key = nn.parameter.Parameter(data=torch.empty((self.pool_size, self.embed_dim)))
        nn.init.uniform_(self.prompt)
        nn.init.uniform_(self.prompt_key)
    
    def forward(self, x_embed: torch.Tensor, cls_feature=None):
        """
        x_embed: (bs, seq_len, embed_dim)
        cls_feature: (bs, embed_dim)
        """
        assert cls_feature.shape[-1] == x_embed.shape[-1]
        if cls_feature:
            prompt_query = cls_feature
        else:
            prompt_query = torch.mean(cls_feature, dim=1)
        
        # make query 
        prompt_key_norm = l2_normalize(self.prompt_key) # (pool_size, embed_dim)
        prompt_query_norm = l2_normalize(prompt_query) # (bs, embed_dim)
        sim = torch.matmul(prompt_key_norm, torch.transpose(prompt_query_norm, 0, 1)) # (pool_size, bs)
        sim = sim.transpose_(0, 1) # (bs, pool_size)
        
        # select prompt and concat
        (sim_top_k, idx) = torch.topk(sim, self.topk, dim=1) # idx: (#bs, )
        batch_prompt_raw = self.prompt[idx, ...] # (bs, prompt_len, embed_dim)
        res = torch.concat((batch_prompt_raw, x_embed))
        
        # Put pull_constraint loss calculation inside
        loss = torch.sum(prompt_key_norm * prompt_query_norm) / prompt_query_norm.shape[0]
        return res, loss

p = PromptModule()
criterion = torch.nn.CrossEntropyLoss()
model = torch.nn.Linear(10, 10)
lamda = 1
for p in model.parameters():
    p.requires_grad = False
def f(x_embed, y, model, use_cls_feature_to_match_prompt=False):
    # x_embed: (bs, seqlen, embed_dim)
    if use_cls_feature_to_match_prompt:
        with torch.no_grad():   
            cls_feature = model.get_cls_feature(x_embed)
        x_embed, prompt_loss = p(x_embed, cls_feature)
    else:
        x_embed, prompt_loss = p(x_embed)
    prompt_len, topk = p.prompt_len, p.topk
    x_embed = model.attention_layers(x_embed)
    x_embed = x_embed[:prompt_len*topk, ...] # (prompt_len*topk, feat_dim)
    x_embed = torch.mean(x_embed, axis=0)
    y_pred = model.cls_layer(x_embed)
    pred_loss = criterion(y_pred, y)
    loss = pred_loss + lamda * prompt_loss
    return loss



