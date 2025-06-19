import torch


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

    # Ending just before 3.4.2

calc_weighted_attn_2()