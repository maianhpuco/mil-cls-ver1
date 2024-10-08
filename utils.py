from tabulate import tabulate
import itertools


def check_layers(model_state_dict, weights_state_dict, header="", align=True):
    matched_layers, discarded_layers = 0, 0

    for name, param in model_state_dict.items():
        if (
            name in weights_state_dict
            and param.size() == weights_state_dict[name].size()
        ):
            matched_layers += 1
        else:
            discarded_layers += 1

    for name, param in weights_state_dict.items():
        if name not in model_state_dict:
            discarded_layers += 1

    print(
        f"{header} >"
        f" Model: {len(model_state_dict.keys())} |"
        f" Weights: {len(weights_state_dict)} |"
        f" Matched: {matched_layers} |"
        f" Discarded: {discarded_layers}"
    )
    if align:
        layers_comparison_table = get_aligned_layers_comparison_table(
            model_state_dict, weights_state_dict, header
        )
    else:
        layers_comparison_table = get_layers_comparison_table(
            model_state_dict, weights_state_dict, header
        )
    print(layers_comparison_table)
    print("\n")


def get_aligned_layers_comparison_table(
    model_state_dict, weights_state_dict, header=""
):
    model_layers = sorted(model_state_dict.keys())
    weight_layers = sorted(weights_state_dict.keys())

    matched_layers = []
    m_ptr = 0
    w_ptr = 0
    for i in range(max(len(model_layers), len(weight_layers))):
        try:
            cur_model_layer = model_layers[m_ptr]
        except IndexError:
            cur_model_layer = ""
        try:
            cur_weight_layer = weight_layers[w_ptr]
        except IndexError:
            cur_weight_layer = ""

        if "" in [cur_model_layer, cur_weight_layer]:
            matched_layers.append((cur_model_layer, cur_weight_layer))
            continue

        if cur_model_layer == cur_weight_layer:
            matched_layers.append((cur_model_layer, cur_weight_layer))
            m_ptr += 1
            w_ptr += 1
        elif cur_model_layer > cur_weight_layer:
            matched_layers.append(("", cur_weight_layer))
            w_ptr += 1
        elif cur_model_layer < cur_weight_layer:
            matched_layers.append((cur_model_layer, ""))
            m_ptr += 1

    table = tabulate(
        matched_layers,
        headers=[f"{header} Model", f"{header} Weights"],
        tablefmt="simple",
    )
    return table


def get_layers_comparison_table(
    model_state_dict, weights_state_dict, header=""
):
    model_layers = sorted(model_state_dict.keys())
    weight_layers = sorted(weights_state_dict.keys())

    matched_layers = list(
        itertools.zip_longest(model_layers, weight_layers, fillvalue="")
    )
    table = tabulate(
        matched_layers,
        headers=[f"{header} Model", f"{header} Weights"],
        tablefmt="simple",
    )
    return table
