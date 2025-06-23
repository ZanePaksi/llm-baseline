import torch
from numpy.ma.core import masked

inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your     (x^1)
         [0.55, 0.87, 0.66],  # journey  (x^2)
         [0.57, 0.85, 0.64],  # starts   (x^3)
         [0.22, 0.58, 0.33],  # with     (x^4)
         [0.77, 0.25, 0.10],  # one      (x^5)
         [0.05, 0.80, 0.55]]  # step     (x^6)
    )


def calc_attn_weights():

    """
        This confused the hell out of me at first but I think i understand.
        with each word in next word prediction attention is calculated
        (how the other words around it affect the next word). So at first, this is fast, less words.
        As we get further into attention we are computing the summed context variables of all the past tokens
    """

    query = inputs[1]
    attn_scores_2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(x_i, query)

    # print(attn_scores_2)
    # attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
    # print(f"{attn_weights_2_tmp=}")
    # print(f"{attn_weights_2_tmp.sum()=}")
    #
    # attn_w_naive_2 = softmax_naive(attn_scores_2)
    # print(f"{attn_w_naive_2=}")
    # print(f"{attn_w_naive_2.sum()=}")
    #
    # attn_w_2 = torch.softmax(attn_scores_2, dim=0)
    # print(f"{attn_w_2=}")
    # print(f"{attn_w_2.sum()=}")

    return attn_scores_2

def softmax_naive(vec):
    return torch.exp(vec) / torch.exp(vec).sum(dim=0)


def calc_context_vec_2():
    query = inputs[1]
    context_vec_2 = torch.zeros(query.shape)
    attn_scores = calc_attn_weights()
    attn_w = torch.softmax(attn_scores, dim=0)
    for i, x_i in enumerate(inputs):
        context_vec_2 += attn_w[i] * x_i
    print(f"{context_vec_2=}")


def calc_all_context_vec():

    # This stage calculates the attention vectors for each embedded token (ET)
    # First pass is an un-optimized solution with for loops
    # attn_scores = torch.empty(6, 6)
    # for i, x_i in enumerate(inputs):
    #     for j, x_j in enumerate(inputs):
    #         attn_scores[i, j] = torch.dot(x_i, x_j)

    # next pass uses the @ symbol, which is supported with numpy, its matrix multiplication
    attn_scores = inputs @ inputs.T
    print(attn_scores)
    attn_weights = torch.softmax(attn_scores, dim=-1)
    print(f"{attn_weights=}")
    print(f"{attn_weights.sum(dim=-1)=}")

    # Now we'll compute all the context vectors via matrix multiplication
    all_context_vecs = attn_weights @ inputs
    print(f"{all_context_vecs=}")

"""
    Above focused entirely on calculating attention scores for then calculating context vectors (3.3)
    Below will focus on implementing trainable weights (3.4)
"""

def calc_weighted_attn_2():
    # Weighted calc on only the second embedded token
    x_2 = inputs[1]
    d_in = inputs.shape[1]
    d_out = 2

    torch.manual_seed(123)
    w_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    w_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    w_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    query_2 = x_2 @ w_query
    key_2 = x_2 @ w_key
    value_2 = x_2 @ w_value
    print(query_2)

    keys = inputs @ w_key
    values = inputs @ w_value
    print(f"{keys.shape=}")
    print(f"{values.shape=}")

    # computing unnormalized attention score for element 2 (x_2)
    keys_2 = keys[1]
    attn_score_22 = query_2.dot(keys_2)
    print(f"{attn_score_22=}")

    # generalize the computation ot all attention scores via matrix multiplication
    attn_scores_2 = query_2 @ keys.T
    print(f"{attn_scores_2}")

    # now moving from attention scores to attention weights
    d_k = keys.shape[-1]
    attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
    print(f"{attn_weights_2=}")


class SelfAttentionV1(torch.nn.Module):
    """
        Section 3.4.2
        d_in and d_out represent the input embedding size
        I.E The dimensionality of the vectors
    """

    def __init__(self, d_in, d_out):
        super().__init__()
        self.w_query = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.w_key = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.w_value = torch.nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, embedded_tokens):
        keys = embedded_tokens @ self.w_key
        queries = embedded_tokens @ self.w_query
        values = embedded_tokens @ self.w_value
        # The book comments attn_scores with 'omega'
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec


def try_SelfAttentionV1():
    d_in = inputs.shape[1]
    d_out = 2
    torch.manual_seed(123)
    sa_v1 = SelfAttentionV1(d_in, d_out)
    print(sa_v1(inputs))


class SelfAttentionV2(torch.nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.w_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, embedded_tokens):
        keys = self.w_key(embedded_tokens)
        queries = self.w_query(embedded_tokens)
        values = self.w_value(embedded_tokens)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec


def try_SelfAttentionV2():
    d_in = inputs.shape[1]
    d_out = 2
    torch.manual_seed(789)
    sa_v2 = SelfAttentionV2(d_in, d_out)
    return sa_v2


def causal_attention_start():
    sa_v2 = try_SelfAttentionV2()
    queries = sa_v2.w_query(inputs)
    keys = sa_v2.w_key(inputs)
    attn_scores = queries @ keys.T
    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    print(attn_weights)

    context_length = attn_scores.shape[0]
    mask_simple = torch.tril(torch.ones(context_length, context_length))
    print(mask_simple)

    mask_simple = attn_weights * mask_simple
    print(mask_simple)

    row_sums = mask_simple.sum(dim=-1, keepdim=True)
    masked_simple_norm = mask_simple / row_sums
    print(masked_simple_norm)

    print(f"\n masking to negative infinity")
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
    print(masked)

    attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
    print(attn_weights)

    # applying dropout to attn_weights
    torch.manual_seed(123)
    dropout = torch.nn.Dropout(0.5)
    print(dropout(attn_weights))


def causal_attention_dropout_example():
    torch.manual_seed(123)
    dropout = torch.nn.Dropout(0.5)
    example = torch.ones(6, 6)
    print(dropout(example))


class CausalAttention(torch.nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.w_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, embedded_tokens):
        b, num_tokens, d_in = embedded_tokens.shape
        keys = self.w_key(embedded_tokens)
        queries = self.w_query(embedded_tokens)
        values = self.w_value(embedded_tokens)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec


def causal_attention_implementation():
    d_in = inputs.shape[1]
    d_out = 2
    batch = torch.stack((inputs, inputs), dim=0)
    torch.manual_seed(123)
    context_length = batch.shape[1]
    ca = CausalAttention(d_in, d_out, context_length, 0.0)
    context_vecs = ca(batch)
    print("context_vecs.shape:", context_vecs.shape)


class MultiHeadAttentionWrapper(torch.nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)]
        )

    def forward(self, embedded_tokens):
        return torch.cat([head(embedded_tokens) for head in self.heads], dim=-1)


def multi_head_wrapper():
    batch = torch.stack((inputs, inputs), dim=0)
    torch.manual_seed(123)
    context_length = batch.shape[1]
    d_in, d_out = 3, 2
    mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)

    print(context_vecs)
    print(f"{context_vecs.shape=}")


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.w_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(d_out, d_out)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, embedded_tokens):
        b, num_tokens, d_in = embedded_tokens.shape

        keys = self.w_key(embedded_tokens)
        queries = self.w_query(embedded_tokens)
        values = self.w_value(embedded_tokens)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # set the dropout values to negative infinity
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # normalize attention tensor with softmax
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


def multi_head_implementation():
    batch = torch.stack((inputs, inputs), dim=0)
    torch.manual_seed(123)
    batch_size, context_length, d_in = batch.shape
    d_out = 2
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)

    print(context_vecs)
    print(f"{context_vecs.shape=}")


def gpt2_initialization():
    """
        These are the specs for the gpt2 model.
    """
    context_length = 1024
    d_in, d_out = 768, 768
    num_heads = 12
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads)