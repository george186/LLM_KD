import torch
import torch.nn as nn
import torch.nn.functional as F


class ClientCrossAttentionAdapter(nn.Module):
    """
    标准化 Cross-Attention Adapter
    功能: 将 Client (OPT) 特征对齐到 Server (Llama) 的语义空间和序列长度
    """
    def __init__(self, client_dim, server_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.server_dim = server_dim
        
        # 1. 维度对齐 (Client -> Server)
        self.input_proj = nn.Linear(client_dim, server_dim)
        
        # 2. Cross Attention (Query=Server, Key/Value=Client)
        self.attn_norm = nn.LayerNorm(server_dim)
        self.attn = nn.MultiheadAttention(embed_dim=server_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # 3. FFN
        self.ffn_norm = nn.LayerNorm(server_dim)
        self.ffn = nn.Sequential(
            nn.Linear(server_dim, server_dim * 4),
            nn.GELU(),
            nn.Linear(server_dim * 4, server_dim),
            nn.Dropout(dropout)
        )
        self.out_norm = nn.LayerNorm(server_dim)

    def forward(self, client_feat, server_query, client_mask=None):
        if server_query.dtype == torch.float16:
            server_query = server_query.to(torch.float32)
        # A. 维度投影
        kv = self.input_proj(client_feat) 
        
        # B. Cross Attention
        q = self.attn_norm(server_query)
        k = v = self.attn_norm(kv)

        # ======== 核心改进：引入 Key Padding Mask ========
        if client_mask is not None:
            # PyTorch 的 key_padding_mask 要求: Padding 位置为 True, 有效位置为 False
            # 你的 mask 中 1 是有效, 0 是 pad，所以需要取反
            key_padding_mask = (client_mask == 0)
        else:
            key_padding_mask = None
        
        attn_out, _ = self.attn(query=q, key=k, value=v, need_weights=False, key_padding_mask=key_padding_mask)
        x = attn_out + server_query # 引入 Server Query 残差
        
        # C. FFN
        x = x + self.ffn(self.ffn_norm(x))
        # x = x + self.ffn(x)
        return self.out_norm(x)

###server
class ServerCrossAttentionAdapter(nn.Module):
    """
    标准化 Cross-Attention Adapter
    功能: 将 Client (OPT) 特征对齐到 Server (Llama) 的语义空间和序列长度
    """
    def __init__(self, client_dim, server_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.server_dim = server_dim
        
        # 1. 维度对齐 (Client -> Server)
        self.input_proj = nn.Linear(client_dim, server_dim)
        
        # 2. Cross Attention (Query=Server, Key/Value=Client)
        self.attn_norm = nn.LayerNorm(server_dim)
        self.attn = nn.MultiheadAttention(embed_dim=server_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # 3. FFN
        self.ffn_norm = nn.LayerNorm(server_dim)
        self.ffn = nn.Sequential(
            nn.Linear(server_dim, server_dim * 4),
            nn.GELU(),
            nn.Linear(server_dim * 4, server_dim),
            nn.Dropout(dropout)
        )
        self.out_norm = nn.LayerNorm(server_dim)

    def forward(self, client_feat, server_query, client_mask=None):
        if client_feat.dtype == torch.float16:
            client_feat = client_feat.to(torch.float32)
        # A. 维度投影
        kv = self.input_proj(client_feat) 
        
        # B. Cross Attention
        q = self.attn_norm(server_query)
        k = v = self.attn_norm(kv)

        # ======== 核心改进：引入 Key Padding Mask ========
        if client_mask is not None:
            # PyTorch 的 key_padding_mask 要求: Padding 位置为 True, 有效位置为 False
            # 你的 mask 中 1 是有效, 0 是 pad，所以需要取反
            key_padding_mask = (client_mask == 0)
        else:
            key_padding_mask = None
        
        attn_out, _ = self.attn(query=q, key=k, value=v, need_weights=False, key_padding_mask=key_padding_mask)
        x = attn_out + server_query # 引入 Server Query 残差

        # x = self.ffn_norm(x) + q # 引入 Server Query 残差
        
        # C. FFN
        x = x + self.ffn(self.ffn_norm(x))
        # x = x + self.ffn(x)
        return self.out_norm(x)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class CrossAttentionAdapter(nn.Module):
    """
    终极泛用型轻量化翻译官 (ReZero + SwiGLU Bottleneck + RMSNorm)
    - Up-Adapter: kv_dim=Client, q_dim=Server
    - Down-Adapter: kv_dim=Server, q_dim=Client
    """
    def __init__(self, kv_dim, q_dim, num_heads=4, dropout=0.1, bottleneck_ratio=4):
        super().__init__()
        self.q_dim = q_dim
        # 确保瓶颈层不要太小，最低 64 维兜底
        self.bottleneck_dim = max(q_dim // bottleneck_ratio, 64) 
        
        # 1. 维度对齐 (将源知识映射到目标空间)
        self.input_proj = nn.Linear(kv_dim, q_dim, bias=False)
        
        # 2. Cross Attention
        self.attn_norm = RMSNorm(q_dim)
        self.attn = nn.MultiheadAttention(embed_dim=q_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        # ReZero 门控机制，初始化为全 0
        self.attn_gate = nn.Parameter(torch.zeros(1)) 
        
        # 3. SwiGLU 瓶颈 FFN
        self.ffn_norm = RMSNorm(q_dim)
        self.ffn_gate_proj = nn.Linear(q_dim, self.bottleneck_dim, bias=False)
        self.ffn_up_proj = nn.Linear(q_dim, self.bottleneck_dim, bias=False)
        self.ffn_down_proj = nn.Linear(self.bottleneck_dim, q_dim, bias=False)
        self.ffn_gate = nn.Parameter(torch.zeros(1))

    def forward(self, kv_feat, q_feat, kv_mask=None):
        if q_feat.dtype == torch.float16:
            q_feat = q_feat.to(torch.float32)
        if kv_feat.dtype == torch.float16:
            kv_feat = kv_feat.to(torch.float32)
            
        # A. 维度映射
        kv = self.input_proj(kv_feat) 
        
        # B. Cross-Attention
        q = self.attn_norm(q_feat)
        k = v = self.attn_norm(kv)
        
        key_padding_mask = (kv_mask == 0) if kv_mask is not None else None
        
        attn_out, _ = self.attn(query=q, key=k, value=v, key_padding_mask=key_padding_mask, need_weights=False)
        
        # 门控残差注入
        x = q_feat + self.attn_gate * attn_out 
        
        # C. SwiGLU FFN (带残差)
        residual = x
        x = self.ffn_norm(x)
        # SwiGLU 公式: Down( SiLU(Gate(x)) * Up(x) )
        x = self.ffn_down_proj(F.silu(self.ffn_gate_proj(x)) * self.ffn_up_proj(x))
        x = residual + self.ffn_gate * x 
        
        return x