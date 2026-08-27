import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Categorical embedding
# ============================================================

class InputEmbeddings(nn.Module):
    """
    Embedding layer for categorical nuclear features.

    Each categorical feature has its own embedding table.

    Example categorical features:
        - Zshell_category
        - Nshell_category
        - ZEO
        - NEO
        - deltaN
        - deltaZ

    Input shape:
        x = (batch, num_categorical_features)

    Output shape:
        embedded = (batch, num_categorical_features, d_model)
    """

    def __init__(self, d_model: int, categorical_feature_sizes: list):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(feature_size, d_model)
            for feature_size in categorical_feature_sizes
        ])

    def forward(self, x):
        embedded_features = [
            embedding(x[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]

        return torch.stack(embedded_features, dim=1)


# ============================================================
# Standard sinusoidal positional encoding
# ============================================================
#
# Important:
# This class is kept for reference, but the main model below does NOT
# use ordinary positional encoding for normal feature tokens.
#
# Reason:
# Nuclear features are not a natural sequence like words in a sentence.
# N, Z, A, shell category, Coulomb term, and symmetry term are fixed
# physical descriptors.
#
# Instead, the model uses:
#     1. feature-wise continuous tokenization
#     2. categorical embeddings
#     3. learnable feature identity embeddings
#
# However, we still use ValueBasedPositionalEncoding for shell categories.
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)

        position = torch.arange(
            0,
            seq_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)


# ============================================================
# Continuous feature tokenizer
# ============================================================

class ContinuousFeatureTokenizer(nn.Module):
    """
    Feature-wise tokenizer for continuous nuclear features.

    Instead of using one shared Linear(1, d_model) for all continuous
    features, each feature gets its own learnable weight and bias:

        token_j = x_j * weight_j + bias_j

    This is more suitable for tabular scientific data because different
    continuous features have different physical meanings.

    Example:
        N is not the same type of feature as Coulomb term.
        Z is not the same type of feature as symmetry term.

    Input shape:
        x = (batch, num_continuous_features)

    Output shape:
        tokens = (batch, num_continuous_features, d_model)
    """

    def __init__(self, num_features: int, d_model: int):
        super().__init__()

        self.num_features = num_features
        self.d_model = d_model

        self.weight = nn.Parameter(
            torch.randn(num_features, d_model) * 0.02
        )

        self.bias = nn.Parameter(
            torch.zeros(num_features, d_model)
        )

    def forward(self, x):
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


# ============================================================
# Value-based positional encoding for shell categories
# ============================================================
#
# We keep this for:
#     - Zshell_category
#     - Nshell_category
#
# These shell categories are not just arbitrary labels.
# They encode shell-related structure. The idea is that shell category
# can be treated like a position or distance-like coordinate from shell
# closure / shell center.
#
#
# This is different from ordinary positional encoding:
#
#     ordinary PE:
#         token 0, token 1, token 2, ...
#
#     value-based shell PE:
#         shell category value 0, 1, 2, ...
#
# So the model can learn smooth relationships between shell-category
# values instead of treating them as completely unrelated labels.
# ============================================================

class ValueBasedPositionalEncoding(nn.Module):
    """
    Value-based positional encoding.

    Applied only to:
        - Zshell_category
        - Nshell_category

    Input:
        x shape:
            (batch, 1, d_model)

        value_indices shape:
            (batch, 1)

    Output:
        x + value-based shell encoding
    """

    def __init__(self, d_model: int, max_value: int, dropout: float):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_value + 1, d_model)

        position = torch.arange(
            0,
            max_value + 1,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x, value_indices):
        value_pe = self.pe[value_indices.long()]
        return self.dropout(x + value_pe)

class HybridContinuousFeatureTokenizer(nn.Module):
    """
    Hybrid tokenizer for continuous nuclear features.

    This combines the old stable representation with a small
    feature-specific correction.

    Old stable part:
        shared_token_j = shared Linear(1 -> d_model)(x_j)

    New feature-specific part:
        feature_token_j = x_j * feature_weight_j + feature_bias_j

    Final:
        token_j = shared_token_j + feature_token_j

    The feature-specific part is initialized near zero, so at the
    beginning of training this behaves almost like the old model.
    During training, the model can learn feature-specific corrections.

    This is scientifically useful because N, Z, A, Coulomb term,
    and symmetry term have different physical meanings, but we do not
    destroy the stable representation that already worked well.
    """

    def __init__(self, num_features: int, d_model: int):
        super().__init__()

        self.num_features = num_features
        self.d_model = d_model

        # Old stable shared projection
        self.shared_proj = nn.Linear(1, d_model)

        # Small feature-specific correction
        self.feature_weight = nn.Parameter(
            torch.zeros(num_features, d_model)
        )

        self.feature_bias = nn.Parameter(
            torch.zeros(num_features, d_model)
        )

    def forward(self, x):
        """
        x shape:
            (batch, num_features)

        output shape:
            (batch, num_features, d_model)
        """

        x_unsqueezed = x.unsqueeze(-1)

        shared_tokens = self.shared_proj(x_unsqueezed)

        feature_correction = (
            x_unsqueezed * self.feature_weight.unsqueeze(0)
            + self.feature_bias.unsqueeze(0)
        )

        return shared_tokens + feature_correction

# ============================================================
# Feed-forward block
# ============================================================

class FeedForwardBlock(nn.Module):
    """
    Standard Transformer feed-forward block.

    Used inside the encoder and also for the fusion layer.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Attention pooling
# ============================================================

class AttentionPooling(nn.Module):
    """
    Learnable attention pooling over feature tokens.

    For each token, the model learns one scalar score:

        score_i = Linear(token_i)

    Then softmax converts scores into weights:

        weight_i = softmax(score_i)

    Final pooled vector:

        pooled = sum_i weight_i * token_i

    This lets the model learn which feature tokens matter more for
    each nucleus.
    """

    def __init__(self, d_model: int):
        super().__init__()

        self.score_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        scores = self.score_layer(x).squeeze(-1)

        weights = torch.softmax(scores, dim=1)

        pooled = torch.sum(
            x * weights.unsqueeze(-1),
            dim=1
        )

        return pooled, weights


# ============================================================
# Gated attention pooling
# ============================================================

class GatedAttentionPooling(nn.Module):
    """
    Gated attention pooling.

    This combines stable mean pooling with flexible attention pooling:

        pooled = (1 - gate) * mean_pool + gate * attention_pool

    The gate is learnable.

    init_gate = -2.0 means:

        sigmoid(-2.0) ≈ 0.12

    So the model starts close to mean pooling:

        about 88% mean pooling
        about 12% attention pooling

    During training, the model can increase or decrease the gate.
    """

    def __init__(self, d_model: int, init_gate: float = -2.0):
        super().__init__()

        self.attention_pool = AttentionPooling(d_model)

        self.gate_logit = nn.Parameter(
            torch.tensor(float(init_gate))
        )

    def forward(self, x):
        mean_pooled = x.mean(dim=1)

        attention_pooled, weights = self.attention_pool(x)

        gate = torch.sigmoid(self.gate_logit)

        pooled = (1.0 - gate) * mean_pooled + gate * attention_pooled

        return pooled, weights, gate


# ============================================================
# Multi-head self-attention block
# ============================================================

class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-head self-attention for feature tokens.

    Input:
        q, k, v shape:
            (batch, num_tokens, d_model)

    Output:
        out shape:
            (batch, num_tokens, d_model)

        attn shape:
            (batch, num_heads, num_tokens, num_tokens)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()

        assert d_model % num_heads == 0, (
            "d_model must be divisible by num_heads."
        )

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]

        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        q = q.view(
            batch_size,
            -1,
            self.num_heads,
            self.d_k
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            -1,
            self.num_heads,
            self.d_k
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            -1,
            self.num_heads,
            self.d_k
        ).transpose(1, 2)

        scores = torch.matmul(
            q,
            k.transpose(-2, -1)
        ) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(
            batch_size,
            -1,
            self.num_heads * self.d_k
        )

        out = self.w_o(out)

        return out, attn


# ============================================================
# Transformer encoder block
# ============================================================

class EncoderBlock(nn.Module):
    """
    Pre-norm style encoder block.

    The structure is:

        x = x + attention(norm(x))
        x = x + ffn(norm(x))

    This is usually more stable than post-norm for deeper Transformers.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttentionBlock(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardBlock(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_input = self.norm1(x)

        attn_output, attn_weights = self.attention(
            attn_input,
            attn_input,
            attn_input,
            mask,
        )

        x = x + self.dropout(attn_output)

        ffn_input = self.norm2(x)
        x = x + self.dropout(self.ffn(ffn_input))

        return x, attn_weights


# ============================================================
# Transformer encoder
# ============================================================

class TransformerEncoder(nn.Module):
    """
    Stack of Transformer encoder blocks.

    The model stores attention weights from each layer so you can later
    visualize feature-feature attention.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()

        self.last_attention_weights = []

        self.layers = nn.ModuleList([
            EncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        self.last_attention_weights = []

        for layer in self.layers:
            x, attn = layer(x, mask)

            # Detach attention weights so they are saved for plotting
            # without keeping the computation graph.
            self.last_attention_weights.append(attn.detach())

        out = self.norm(x)

        return out, self.last_attention_weights

    def get_attention_weights(self):
        return self.last_attention_weights


# ============================================================
# Transformer model for nuclear mass excess regression
# ============================================================

class TransformerMassExcessPredictor(nn.Module):
    """
    Transformer model for nuclear mass excess prediction.

    Main ideas:

    1. Continuous features use feature-wise tokenization.
       Each continuous feature has its own learnable weight and bias.

    2. Categorical features use embedding tables.

    3. Zshell_category and Nshell_category keep your physics-motivated
       ValueBasedPositionalEncoding.

       This treats the shell-category value like a shell-position or
       shell-distance coordinate.

    4. Ordinary sinusoidal positional encoding is NOT applied to all
       feature tokens, because the input is not a natural sequence.

    5. Learnable feature identity embeddings tell the model which token
       corresponds to which physical feature.

    6. Pooling can be:
          - mean
          - attention
          - gated_attention
    """

    def __init__(
        self,
        num_cont_features,
        categorical_feature_sizes,
        categorical_features,
        d_model=128,
        num_heads=8,
        d_ff=512,
        num_layers=4,
        dropout=0.10,
        pooling_type="mean",
        gate_init=-2.0,
    ):
        super().__init__()

        self.num_cont_features = num_cont_features
        self.num_cat_features = len(categorical_feature_sizes)
        self.has_categorical = self.num_cat_features > 0
        self.d_model = d_model
        self.categorical_features = categorical_features

        # ------------------------------------------------------------
        # Pooling type
        # ------------------------------------------------------------

        self.pooling_type = pooling_type

        if self.pooling_type not in ["mean", "attention", "gated_attention"]:
            raise ValueError(
                "pooling_type must be 'mean', 'attention', or 'gated_attention'."
            )

        self.cat_attention_pool = AttentionPooling(d_model)
        self.cont_attention_pool = AttentionPooling(d_model)

        self.cat_gated_pool = GatedAttentionPooling(
            d_model=d_model,
            init_gate=gate_init,
        )

        self.cont_gated_pool = GatedAttentionPooling(
            d_model=d_model,
            init_gate=gate_init,
        )

        # ------------------------------------------------------------
        # Categorical tokens
        # ------------------------------------------------------------

        self.shell_pe_feature_names = {
            "Zshell_category",
            "Nshell_category",
        }

        if self.has_categorical:
            assert len(categorical_features) == len(categorical_feature_sizes), (
                "categorical_features and categorical_feature_sizes "
                "must have the same length."
            )

            self.categorical_embedding = InputEmbeddings(
                d_model=d_model,
                categorical_feature_sizes=categorical_feature_sizes,
            )
            
            self.cat_positional_encoding = PositionalEncoding(
                d_model=d_model,
                seq_len=self.num_cat_features,
                dropout=dropout
                )
            

            # Detect shell-category features.
            # Only these features receive ValueBasedPositionalEncoding.
            self.shell_cat_indices = [
                idx
                for idx, name in enumerate(categorical_features)
                if name in self.shell_pe_feature_names
            ]

            self.other_cat_indices = [
                idx
                for idx, name in enumerate(categorical_features)
                if name not in self.shell_pe_feature_names
            ]

            self.shell_value_positional_encodings = nn.ModuleDict()

            for idx in self.shell_cat_indices:
                feature_name = categorical_features[idx]

                self.shell_value_positional_encodings[feature_name] = (
                    ValueBasedPositionalEncoding(
                        d_model=d_model,
                        max_value=categorical_feature_sizes[idx] - 1,
                        dropout=dropout,
                    )
                )

        else:
            self.categorical_embedding = None
            self.shell_cat_indices = []
            self.other_cat_indices = []
            self.shell_value_positional_encodings = nn.ModuleDict()

        # ------------------------------------------------------------
        # Continuous tokens
        # ------------------------------------------------------------
        #
        # This replaces the old shared Linear(1, d_model).
        # ------------------------------------------------------------

        self.cont_tokenizer = HybridContinuousFeatureTokenizer(
            num_features=num_cont_features,
            d_model=d_model
            )

        # ------------------------------------------------------------
        # Learnable feature identity embeddings
        # ------------------------------------------------------------
        #
        # Token order in the current dataloader/model is:
        #
        #     categorical tokens first
        #     continuous tokens second
        #
        # This embedding tells the Transformer which physical feature
        # each token represents.
        #
        # It replaces ordinary positional encoding for tabular features.
        # ------------------------------------------------------------

        self.total_num_tokens = self.num_cat_features + self.num_cont_features

        self.feature_id_embedding = nn.Embedding(
            self.total_num_tokens,
            d_model,
        )

        self.feature_dropout = nn.Dropout(dropout)

        # ------------------------------------------------------------
        # Transformer encoder
        # ------------------------------------------------------------

        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout,
        )

        # ------------------------------------------------------------
        # Regression head
        # ------------------------------------------------------------
        #
        # The model separately pools categorical and continuous branches.
        # Then it concatenates them:
        #
        #     fused = [pooled_cat, pooled_cont]
        #
        # So the fused dimension is 2 * d_model.
        # ------------------------------------------------------------

        self.fusion = FeedForwardBlock(
            d_model=d_model * 2,
            d_ff=d_ff,
            dropout=dropout,
        )

        # Linear baseline from continuous features.
        # This helps the model learn a simple global trend first.
        self.baseline_layer = nn.Linear(num_cont_features, 1)

        # Nonlinear correction from Transformer representation.
        self.regressor = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def _build_categorical_tokens(self, categorical_inputs):
        """
        Build categorical tokens.

        This function is separated from forward() to keep the logic clean.

        Steps:
            1. Embed all categorical features.
            2. Apply ValueBasedPositionalEncoding only to shell categories:
                   Zshell_category
                   Nshell_category

        We do NOT apply ordinary sinusoidal positional encoding here.
        """

        categorical_inputs = categorical_inputs.long()

        cat_tokens = self.categorical_embedding(categorical_inputs)
        
        # Keep ordinary categorical positional encoding as a feature-slot anchor.
        # This helps the model distinguish categorical feature positions.
        # Shell features still get additional value-based shell encoding below.
        cat_tokens = self.cat_positional_encoding(cat_tokens)

        # We clone because we modify selected shell-category tokens.
        cat_tokens = cat_tokens.clone()

        for idx in self.shell_cat_indices:
            feature_name = self.categorical_features[idx]

            shell_token = cat_tokens[:, idx:idx + 1, :]
            shell_value = categorical_inputs[:, idx:idx + 1]

            shell_token = self.shell_value_positional_encodings[feature_name](
                shell_token,
                shell_value,
            )

            cat_tokens[:, idx:idx + 1, :] = shell_token

        return cat_tokens

    def _pool_tokens(
        self,
        encoded_tokens,
        num_cat_tokens,
        batch_size,
        continuous_inputs,
    ):
        """
        Pool encoded feature tokens into one regression vector.

        Categorical and continuous branches are pooled separately because
        they represent different feature types.
        """

        if self.has_categorical:
            encoded_cat = encoded_tokens[:, :num_cat_tokens, :]
            encoded_cont = encoded_tokens[:, num_cat_tokens:, :]

            if self.pooling_type == "mean":
                pooled_cat = encoded_cat.mean(dim=1)
                pooled_cont = encoded_cont.mean(dim=1)

            elif self.pooling_type == "attention":
                pooled_cat, _ = self.cat_attention_pool(encoded_cat)
                pooled_cont, _ = self.cont_attention_pool(encoded_cont)

            elif self.pooling_type == "gated_attention":
                pooled_cat, _, _ = self.cat_gated_pool(encoded_cat)
                pooled_cont, _, _ = self.cont_gated_pool(encoded_cont)

        else:
            pooled_cat = torch.zeros(
                batch_size,
                self.d_model,
                device=continuous_inputs.device,
            )

            if self.pooling_type == "mean":
                pooled_cont = encoded_tokens.mean(dim=1)

            elif self.pooling_type == "attention":
                pooled_cont, _ = self.cont_attention_pool(encoded_tokens)

            elif self.pooling_type == "gated_attention":
                pooled_cont, _, _ = self.cont_gated_pool(encoded_tokens)

        return pooled_cat, pooled_cont

    def forward(self, categorical_inputs, continuous_inputs, mask=None):
        batch_size = continuous_inputs.size(0)

        token_list = []

        # ------------------------------------------------------------
        # 1. Categorical feature tokens
        # ------------------------------------------------------------

        if self.has_categorical:
            cat_tokens = self._build_categorical_tokens(categorical_inputs)
            token_list.append(cat_tokens)
            num_cat_tokens = cat_tokens.size(1)
        else:
            num_cat_tokens = 0

        # ------------------------------------------------------------
        # 2. Continuous feature tokens
        # ------------------------------------------------------------

        cont_tokens = self.cont_tokenizer(continuous_inputs)
        token_list.append(cont_tokens)

        # ------------------------------------------------------------
        # 3. Combine tokens
        # ------------------------------------------------------------

        tokens = torch.cat(token_list, dim=1)

        # ------------------------------------------------------------
        # 4. Add feature identity embeddings
        # ------------------------------------------------------------
        #
        # This is the tabular equivalent of telling the Transformer:
        #
        #     token 0 means Zshell_category
        #     token 1 means Nshell_category
        #     ...
        #
        # The exact mapping follows your dataloader/model order:
        # categorical tokens first, then continuous tokens.
        # ------------------------------------------------------------

        token_ids = torch.arange(
            tokens.size(1),
            device=tokens.device,
        ).unsqueeze(0).expand(tokens.size(0), -1)

        tokens = tokens + self.feature_id_embedding(token_ids)
        tokens = self.feature_dropout(tokens)

        # ------------------------------------------------------------
        # 5. Transformer encoder
        # ------------------------------------------------------------

        encoded_tokens, attention_weights = self.encoder(tokens, mask)

        # ------------------------------------------------------------
        # 6. Pool encoded tokens
        # ------------------------------------------------------------

        pooled_cat, pooled_cont = self._pool_tokens(
            encoded_tokens=encoded_tokens,
            num_cat_tokens=num_cat_tokens,
            batch_size=batch_size,
            continuous_inputs=continuous_inputs,
        )

        # ------------------------------------------------------------
        # 7. Regression
        # ------------------------------------------------------------

        fused = torch.cat([pooled_cat, pooled_cont], dim=1)

        fused = self.fusion(fused)

        baseline = self.baseline_layer(continuous_inputs).squeeze(1)

        correction = self.regressor(fused).squeeze(1)

        predicted_mass_excess = baseline + correction

        return predicted_mass_excess, attention_weights