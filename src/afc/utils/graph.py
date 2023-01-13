import torch
import torch_geometric as geom
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from enum import Enum
from typing import Union, List

__graph_topo_cache = {}
CONNECTIVITY_OPTIONS = [
    "full",
    "local_plus_global",
    "local",
    "chain",
    "loop",
    "hierarchical_aggr",
    "none",
]


class ConnectivityOption(Enum):
    FULL = 1
    FORWARD_BACKWARD = 2
    CHAIN = 3
    LOOP = 4
    HIERARCHICAL_AGGR = 5
    NONE = 6


def positional_encoding(pos: int, dim: int, dev: str = None):
    assert dim % 2 == 0
    
    def omega(k):
        return 1.0 / np.power(10000, 2 * k / dim)
    
    encoding = torch.tensor([
        np.sin(omega(np.floor(i / 2)) * pos) if i % 2 == 0
        else np.cos(omega(np.floor(i / 2)) * pos)
        for i in range(dim)
    ])

    if dev:
        encoding = encoding.to(dev)
    
    return encoding


def create_positional_encoding(length: int, dim: int, dev: str = None):
    enc = torch.concat([ positional_encoding(pos, dim)[None, :] for pos in range(length) ], dim=0)
    enc = enc.to(dev)

    return enc


def create_hierarchical_aggr_topo(n_nodes: int, group_size: int):
    # Create the chain edges
    chain_edges = torch.arange(n_nodes - 1).repeat(2, 1)
    chain_edges[1, :] += 1

    # Create the first layer aggregation edges
    n_aggr_nodes = int(np.ceil(n_nodes / group_size))
    aggr_edges = torch.stack(
        (
            torch.arange(n_aggr_nodes * group_size),
            torch.arange(n_nodes, n_nodes + n_aggr_nodes).repeat_interleave(group_size),
        )
    )[:, :n_nodes]

    # Create the top node edges
    top_node_id = n_nodes + n_aggr_nodes
    top_node_edges = torch.stack(
        (
            torch.arange(n_nodes, n_nodes + n_aggr_nodes),
            torch.tensor([top_node_id]).repeat(n_aggr_nodes),
        )
    )

    # Concat the edges and add their reverse
    edges = torch.concat((chain_edges, aggr_edges, top_node_edges), dim=1)
    edges = torch.concat((edges, edges[[1, 0], :]), dim=1)

    return edges


def create_graph_topo(
    n_nodes: int,
    connectivity: str = "full",
    device: Union[str, torch.device, None] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Creates a graph's topology (edge list).

    Args:
        n_nodes:
            An integer specifying the number of nodes in the graph.
        connectivity:
            Indicates the pattern based on which the edges are constructed.
        device:
            Device to which the `edge_index` tensor should be moved.
            The tensor is cached on this device as well, reducing the
            number of data movements. This argument is optional.
    """
    assert connectivity in CONNECTIVITY_OPTIONS, "Unknown connectivity setting"

    if (connectivity, n_nodes) in __graph_topo_cache:
        edge_index = __graph_topo_cache[(connectivity, n_nodes)].detach().clone()
        if device:
            edge_index = edge_index.to(device)

        return edge_index

    if connectivity == "local":
        # Connects neighbouring chunks in the text
        edge_index = torch.diag(torch.ones(n_nodes))
        edge_index += torch.diag(torch.ones(n_nodes - 1), 1)
        edge_index += torch.diag(torch.ones(n_nodes - 1), -1)
        edge_index = torch.stack(torch.where(edge_index))
    elif connectivity == "local_plus_global":
        # Connects neighbouring chunks in the text + add a latent node
        edge_index = torch.diag(torch.ones(n_nodes + 1))
        edge_index += torch.diag(torch.ones(n_nodes), 1)
        edge_index += torch.diag(torch.ones(n_nodes), -1)
        edge_index[:, -1] = 1
        edge_index[-1, :] = 1
        edge_index = torch.stack(torch.where(edge_index))
    elif connectivity == "chain" or connectivity == "loop":
        # Connects subsequent chunks in the text
        edge_index = torch.stack(torch.where(torch.diag(torch.ones(n_nodes - 1), 1)))
        # Additionally loops the last chunk back to the first one
        if connectivity == "loop":
            edge_index = torch.cat(
                (edge_index, torch.tensor([[n_nodes - 1], [0]])), dim=1
            )
    elif connectivity == "none":
        # A graph with no connections
        edge_index = torch.tensor([[], []], dtype=torch.long)
    elif connectivity == "hierarchical_aggr":
        edge_index = create_hierarchical_aggr_topo(n_nodes, **kwargs)
    else:
        # Creates a fully connected graph (including self-loops)
        node_ids = torch.arange(n_nodes)
        edge_index = torch.cartesian_prod(node_ids, node_ids).T

    if device is not None:
        edge_index = edge_index.to(device)

    __graph_topo_cache[(connectivity, n_nodes)] = edge_index.detach().clone()

    return edge_index


def convert_to_graph(
    nodes: Union[torch.Tensor, List],
    label: int,
    connectivity: str = "full",
    return_dict: bool = False,
    device: Union[str, torch.device, None] = None,
) -> Union[geom.data.Data, dict]:
    """
    Creates a graph out of the given nodes.
    Meant for creating a graph representing a document after it has
    been split into multiple nodes. The order of the nodes does matter.

    Args:
        nodes:
            The nodes into which the document has been split.
            Represent encoded parts of the document.
        label:
            Label representing the class of the example.
        connectivity:
            Indicates the topology of the resulting graph.
        return_dict:
            Boolean indicating whether to return the data as a
            torch_geometric.data.Data or a Python dictionary.
        device:
            Device to which the `edge_index` tensor should be moved.
            If no device is provided, then `edge_index` is placed onto
            the same device as the `nodes`.
    """
    assert connectivity in CONNECTIVITY_OPTIONS, "Unknown connectivity setting"

    if isinstance(nodes, list):
        nodes = torch.tensor(nodes)

    n_nodes = nodes.shape[0]

    if not device:
        device = nodes.device

    edge_index = create_graph_topo(n_nodes, connectivity, device=device)

    if connectivity == "local_plus_global":
        nodes = torch.vstack((nodes, nodes.mean(dim=0)))

    if return_dict:
        return {
            "x": nodes,
            "edge_index": edge_index,
            "y": int(label),
        }

    return geom.data.Data(
        x=nodes,
        edge_index=edge_index,
        y=label.to(int)
        if isinstance(label, torch.Tensor)
        else torch.tensor(label, dtype=int),
    )


def create_graph_batch_from_nodes(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_idx: torch.Tensor,
    connectivity: str = "full",
) -> geom.data.Batch:
    assert x.shape[0] == torch.numel(
        batch_idx
    ), "The number of nodes does not equal the number of batch assignments"
    assert batch_idx.dim() == 1
    assert (
        torch.numel(y) == batch_idx.max() + 1
    ), f"The number of classes does not equal the batch_size, {y.size()} != {batch_idx.max() + 1}"

    graph_batch = geom.data.Batch.from_data_list(
        [
            convert_to_graph(
                nodes=x[batch_idx == idx, :],
                label=y[idx],
                connectivity=connectivity,
            )
            for idx in range(y.shape[0])
        ]
    )

    return graph_batch


def create_graph_batch_topo(
    batch_idx: torch.Tensor, connectivity: str = "full"
) -> torch.Tensor:
    """Creates topology for individual graphs in the batch so that they can be represented as a single graph."""
    assert batch_idx.dim() == 1, "The `batch_idx` tensor should be flat"

    unique, counts = torch.unique_consecutive(batch_idx, return_counts=True)
    start_idx = torch.zeros(unique.size())
    start_idx[1:] = torch.cumsum(counts, dim=0)[:-1]

    edge_index = torch.concat(
        [
            create_graph_topo(
                int(n_nodes), connectivity=connectivity, device=batch_idx.device
            )
            + idx
            for n_nodes, idx in zip(counts, start_idx)
        ],
        dim=1,
    ).to(torch.long)

    return edge_index


# Still do not understand why this approach is slower for backpropagation
# than the other one.
# Soooo, it's not slower, it's just that the other version incorrectly
# threw away the computational graph by using x = torch.tensor(x).
def create_graph_batch_from_nodes_direct(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_idx: torch.Tensor,
    connectivity: str = "full",
) -> geom.data.Batch:
    assert x.shape[0] == torch.numel(batch_idx)
    assert batch_idx.dim() == 1
    assert torch.numel(y) == batch_idx.max() + 1, f"{y.size()} != {batch_idx.max() + 1}"

    edge_index = create_graph_batch_topo(batch_idx=batch_idx, connectivity=connectivity)
    edge_index = edge_index.to(x.device)

    return geom.data.Batch(x=x, y=y, batch=batch_idx, edge_index=edge_index)
