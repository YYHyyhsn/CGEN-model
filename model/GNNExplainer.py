from math import sqrt

import warnings
from typing import Any, Dict, Optional, Union, Tuple

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from torch_geometric.explain import ExplainerConfig, Explanation, ModelConfig
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel


class GNNExplainer(ExplainerAlgorithm):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and node features that play a crucial role in the predictions
    made by a GNN.

    .. note::

        For an example of using :class:`GNNExplainer`, see
        `examples/explain/gnn_explainer.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/explain/gnn_explainer.py>`_,
        `examples/explain/gnn_explainer_ba_shapes.py <https://github.com/
        pyg-team/pytorch_geometric/blob/master/examples/
        explain/gnn_explainer_ba_shapes.py>`_, and `examples/explain/
        gnn_explainer_link_pred.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/explain/gnn_explainer_link_pred.py>`_.

    .. note::

        The :obj:`edge_size` coefficient is multiplied by the number of nodes
        in the explanation at every iteration, and the resulting value is added
        to the loss as a regularization term, with the goal of producing
        compact explanations.
        A higher value will push the algorithm towards explanations with less
        elements.
        Consider adjusting the :obj:`edge_size` coefficient according to the
        average node degree in the dataset, especially if this value is bigger
        than in the datasets used in the original paper.

    Args:
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        **kwargs (optional): Additional hyper-parameters to override default
            settings in
            :attr:`~torch_geometric.explain.algorithm.GNNExplainer.coeffs`.
    """

    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
        'EPS': 1e-15,
    }

    def __init__(self, epochs: int = 100, lr: float = 0.01, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)

        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")

        self._train(model, x, edge_index, target=target, index=index, **kwargs)

        node_mask = self._post_process_mask(
            self.node_mask,
            self.hard_node_mask,
            apply_sigmoid=True,
        )
        edge_mask = self._post_process_mask(
            self.edge_mask,
            self.hard_edge_mask,
            apply_sigmoid=True,
        )

        self._clean_model(model)

        return Explanation(node_mask=node_mask, edge_mask=edge_mask)

    def supports(self) -> bool:
        return True

    def _train(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):
        self._initialize_masks(x, edge_index)

        parameters = []
        if self.node_mask is not None:
            parameters.append(self.node_mask)
        if self.edge_mask is not None:
            set_masks(model, self.edge_mask, edge_index, apply_sigmoid=True)
            parameters.append(self.edge_mask)

        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        for i in range(self.epochs):
            optimizer.zero_grad()

            h = x if self.node_mask is None else x * self.node_mask.sigmoid()
            y_hat, y = model(h, edge_index, **kwargs), target

            if index is not None:
                y_hat, y = y_hat[index], y[index]

            loss = self._loss(y_hat, y)

            loss.backward()
            optimizer.step()

            # In the first iteration, we collect the nodes and edges that are
            # involved into making the prediction. These are all the nodes and
            # edges with gradient != 0 (without regularization applied).
            if i == 0 and self.node_mask is not None:
                if self.node_mask.grad is None:
                    raise ValueError("Could not compute gradients for node "
                                     "features. Please make sure that node "
                                     "features are used inside the model or "
                                     "disable it via `node_mask_type=None`.")
                self.hard_node_mask = self.node_mask.grad != 0.0
            if i == 0 and self.edge_mask is not None:
                if self.edge_mask.grad is None:
                    raise ValueError("Could not compute gradients for edges. "
                                     "Please make sure that edges are used "
                                     "via message passing inside the model or "
                                     "disable it via `edge_mask_type=None`.")
                self.hard_edge_mask = self.edge_mask.grad != 0.0

    def _initialize_masks(self, x: Tensor, edge_index: Tensor):
        node_mask_type = self.explainer_config.node_mask_type
        edge_mask_type = self.explainer_config.edge_mask_type

        device = x.device
        (N, F), E = x.size(), edge_index.size(1)

        std = 0.1
        if node_mask_type is None:
            self.node_mask = None
        elif node_mask_type == MaskType.object:
            self.node_mask = Parameter(torch.randn(N, 1, device=device) * std)
        elif node_mask_type == MaskType.attributes:
            self.node_mask = Parameter(torch.randn(N, F, device=device) * std)
        elif node_mask_type == MaskType.common_attributes:
            self.node_mask = Parameter(torch.randn(1, F, device=device) * std)
        else:
            assert False

        if edge_mask_type is None:
            self.edge_mask = None
        elif edge_mask_type == MaskType.object:
            std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
            self.edge_mask = Parameter(torch.randn(E, device=device) * std)
        else:
            assert False

    def _loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        if self.model_config.mode == ModelMode.binary_classification:
            loss = self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y)
        else:
            assert False

        if self.hard_edge_mask is not None:
            assert self.edge_mask is not None
            m = self.edge_mask[self.hard_edge_mask].sigmoid()
            edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
            loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
            ent = -m * torch.log(m + self.coeffs['EPS']) - (
                1 - m) * torch.log(1 - m + self.coeffs['EPS'])
            loss = loss + self.coeffs['edge_ent'] * ent.mean()

        if self.hard_node_mask is not None:
            assert self.node_mask is not None
            m = self.node_mask[self.hard_node_mask].sigmoid()
            node_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
            loss = loss + self.coeffs['node_feat_size'] * node_reduce(m)
            ent = -m * torch.log(m + self.coeffs['EPS']) - (
                1 - m) * torch.log(1 - m + self.coeffs['EPS'])
            loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def _clean_model(self, model):
        clear_masks(model)
        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None


class GNNExplainer_:
    r"""Deprecated version for :class:`GNNExplainer`."""

    coeffs = GNNExplainer.coeffs

    conversion_node_mask_type = {
        'feature': 'common_attributes',
        'individual_feature': 'attributes',
        'scalar': 'object',
    }

    conversion_return_type = {
        'log_prob': 'log_probs',
        'prob': 'probs',
        'raw': 'raw',
        'regression': 'raw',
    }

    def __init__(
        self,
        model: torch.nn.Module,
        epochs: int = 100,
        lr: float = 0.01,
        return_type: str = 'log_prob',
        feat_mask_type: str = 'feature',
        allow_edge_mask: bool = True,
        **kwargs,
    ):
        assert feat_mask_type in ['feature', 'individual_feature', 'scalar']

        explainer_config = ExplainerConfig(
            explanation_type='model',
            node_mask_type=self.conversion_node_mask_type[feat_mask_type],
            edge_mask_type=MaskType.object if allow_edge_mask else None,
        )
        model_config = ModelConfig(
            mode='regression'
            if return_type == 'regression' else 'multiclass_classification',
            task_level=ModelTaskLevel.node,
            return_type=self.conversion_return_type[return_type],
        )

        self.model = model
        self._explainer = GNNExplainer(epochs=epochs, lr=lr, **kwargs)
        self._explainer.connect(explainer_config, model_config)

    @torch.no_grad()
    def get_initial_prediction(self, *args, **kwargs) -> Tensor:

        training = self.model.training
        self.model.eval()

        out = self.model(*args, **kwargs)
        if (self._explainer.model_config.mode ==
                ModelMode.multiclass_classification):
            out = out.argmax(dim=-1)

        self.model.train(training)

        return out

    def explain_graph(
        self,
        x: Tensor,
        edge_index: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        self._explainer.model_config.task_level = ModelTaskLevel.graph

        explanation = self._explainer(
            self.model,
            x,
            edge_index,
            target=self.get_initial_prediction(x, edge_index, **kwargs),
            **kwargs,
        )
        return self._convert_output(explanation, edge_index)

    def explain_node(
        self,
        node_idx: int,
        x: Tensor,
        edge_index: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        self._explainer.model_config.task_level = ModelTaskLevel.node
        explanation = self._explainer(
            self.model,
            x,
            edge_index,
            target=self.get_initial_prediction(x, edge_index, **kwargs),
            index=node_idx,
            **kwargs,
        )
        return self._convert_output(explanation, edge_index, index=node_idx,
                                    x=x)

    def _convert_output(self, explanation, edge_index, index=None, x=None):
        node_mask = explanation.get('node_mask')
        edge_mask = explanation.get('edge_mask')

        if node_mask is not None:
            node_mask_type = self._explainer.explainer_config.node_mask_type
            if node_mask_type in {MaskType.object, MaskType.common_attributes}:
                node_mask = node_mask.view(-1)

        if edge_mask is None:
            if index is not None:
                _, edge_mask = self._explainer._get_hard_masks(
                    self.model, index, edge_index, num_nodes=x.size(0))
                edge_mask = edge_mask.to(x.dtype)
            else:
                edge_mask = torch.ones(edge_index.size(1),
                                       device=edge_index.device)

        return node_mask, edge_mask



from torch_geometric.explain import (
    ExplainerAlgorithm,
    Explanation,
    HeteroExplanation,
)
from torch_geometric.explain.algorithm.utils import (
    clear_masks,
    set_hetero_masks,
    set_masks,
)
from torch_geometric.explain.config import (
    ExplainerConfig,
    ExplanationType,
    MaskType,
    ModelConfig,
    ModelMode,
    ModelReturnType,
    ThresholdConfig,
)
from torch_geometric.typing import EdgeType, NodeType


class Explainer:
    r"""An explainer class for instance-level explanations of Graph Neural
    Networks.

    Args:
        model (torch.nn.Module): The model to explain.
        algorithm (ExplainerAlgorithm): The explanation algorithm.
        explanation_type (ExplanationType or str): The type of explanation to
            compute. The possible values are:

                - :obj:`"model"`: Explains the model prediction.

                - :obj:`"phenomenon"`: Explains the phenomenon that the model
                  is trying to predict.

            In practice, this means that the explanation algorithm will either
            compute their losses with respect to the model output
            (:obj:`"model"`) or the target output (:obj:`"phenomenon"`).
        model_config (ModelConfig): The model configuration.
            See :class:`~torch_geometric.explain.config.ModelConfig` for
            available options. (default: :obj:`None`)
        node_mask_type (MaskType or str, optional): The type of mask to apply
            on nodes. The possible values are (default: :obj:`None`):

                - :obj:`None`: Will not apply any mask on nodes.

                - :obj:`"object"`: Will mask each node.

                - :obj:`"common_attributes"`: Will mask each feature.

                - :obj:`"attributes"`: Will mask each feature across all nodes.

        edge_mask_type (MaskType or str, optional): The type of mask to apply
            on edges. Has the sample possible values as :obj:`node_mask_type`.
            (default: :obj:`None`)
        threshold_config (ThresholdConfig, optional): The threshold
            configuration.
            See :class:`~torch_geometric.explain.config.ThresholdConfig` for
            available options. (default: :obj:`None`)
    """
    def __init__(
        self,
        model: torch.nn.Module,
        algorithm: ExplainerAlgorithm,
        explanation_type: Union[ExplanationType, str],
        model_config: Union[ModelConfig, Dict[str, Any]],
        node_mask_type: Optional[Union[MaskType, str]] = None,
        edge_mask_type: Optional[Union[MaskType, str]] = None,
        threshold_config: Optional[ThresholdConfig] = None,
    ):
        explainer_config = ExplainerConfig(
            explanation_type=explanation_type,
            node_mask_type=node_mask_type,
            edge_mask_type=edge_mask_type,
        )

        self.model = model
        self.algorithm = algorithm

        self.explanation_type = explainer_config.explanation_type
        self.model_config = ModelConfig.cast(model_config)
        self.node_mask_type = explainer_config.node_mask_type
        self.edge_mask_type = explainer_config.edge_mask_type
        self.threshold_config = ThresholdConfig.cast(threshold_config)

        self.algorithm.connect(explainer_config, self.model_config)

    @torch.no_grad()
    def get_prediction(self, *args, **kwargs) -> Tensor:
        r"""Returns the prediction of the model on the input graph.

        If the model mode is :obj:`"regression"`, the prediction is returned as
        a scalar value.
        If the model mode is :obj:`"multiclass_classification"` or
        :obj:`"binary_classification"`, the prediction is returned as the
        predicted class label.

        Args:
            *args: Arguments passed to the model.
            **kwargs (optional): Additional keyword arguments passed to the
                model.
        """
        training = self.model.training
        self.model.eval()

        with torch.no_grad():
            out = self.model(*args, **kwargs)

        self.model.train(training)

        return out

    def get_masked_prediction(
        self,
        x: Union[Tensor, Dict[NodeType, Tensor]],
        edge_index: Union[Tensor, Dict[EdgeType, Tensor]],
        node_mask: Optional[Union[Tensor, Dict[NodeType, Tensor]]] = None,
        edge_mask: Optional[Union[Tensor, Dict[EdgeType, Tensor]]] = None,
        **kwargs,
    ) -> Tensor:
        r"""Returns the prediction of the model on the input graph with node
        and edge masks applied.
        """
        if isinstance(x, Tensor) and node_mask is not None:
            x = node_mask * x
        elif isinstance(x, dict) and node_mask is not None:
            x = {key: value * node_mask[key] for key, value in x.items()}

        if isinstance(edge_mask, Tensor):
            set_masks(self.model, edge_mask, edge_index, apply_sigmoid=False)
        elif isinstance(edge_mask, dict):
            set_hetero_masks(self.model, edge_mask, edge_index,
                             apply_sigmoid=False)

        out = self.get_prediction(x, edge_index, **kwargs)
        clear_masks(self.model)
        return out

    def __call__(
        self,
        x: Union[Tensor, Dict[NodeType, Tensor]],
        edge_index: Union[Tensor, Dict[EdgeType, Tensor]],
        *,
        target: Optional[Tensor] = None,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Union[Explanation, HeteroExplanation]:
        r"""Computes the explanation of the GNN for the given inputs and
        target.

        .. note::

            If you get an error message like "Trying to backward through the
            graph a second time", make sure that the target you provided
            was computed with :meth:`torch.no_grad`.

        Args:
            x (Union[torch.Tensor, Dict[NodeType, torch.Tensor]]): The input
                node features of a homogeneous or heterogeneous graph.
            edge_index (Union[torch.Tensor, Dict[NodeType, torch.Tensor]]): The
                input edge indices of a homogeneous or heterogeneous graph.
            target (torch.Tensor): The target of the model.
                If the explanation type is :obj:`"phenomenon"`, the target has
                to be provided.
                If the explanation type is :obj:`"model"`, the target should be
                set to :obj:`None` and will get automatically inferred. For
                classification tasks, the target needs to contain the class
                labels. (default: :obj:`None`)
            index (Union[int, Tensor], optional): The indices in the
                first-dimension of the model output to explain.
                Can be a single index or a tensor of indices.
                If set to :obj:`None`, all model outputs will be explained.
                (default: :obj:`None`)
            **kwargs: additional arguments to pass to the GNN.
        """
        # Choose the `target` depending on the explanation type:
        prediction: Optional[Tensor] = None
        if self.explanation_type == ExplanationType.phenomenon:
            if target is None:
                raise ValueError(
                    f"The 'target' has to be provided for the explanation "
                    f"type '{self.explanation_type.value}'")
        elif self.explanation_type == ExplanationType.model:
            if target is not None:
                warnings.warn(
                    f"The 'target' should not be provided for the explanation "
                    f"type '{self.explanation_type.value}'")
            prediction = self.get_prediction(x, edge_index, **kwargs)
            target = self.get_target(prediction)

        if isinstance(index, int):
            index = torch.tensor([index])

        training = self.model.training
        self.model.eval()

        explanation = self.algorithm(
            self.model,
            x,
            edge_index,
            target=target,
            index=index,
            **kwargs,
        )

        self.model.train(training)

        # Add explainer objectives to the `Explanation` object:
        explanation._model_config = self.model_config
        explanation.prediction = prediction
        explanation.target = target
        explanation.index = index

        # Add model inputs to the `Explanation` object:
        if isinstance(explanation, Explanation):
            explanation._model_args = list(kwargs.keys())
            explanation.x = x
            explanation.edge_index = edge_index

            for key, arg in kwargs.items():  # Add remaining `kwargs`:
                explanation[key] = arg

        elif isinstance(explanation, HeteroExplanation):
            # TODO Add `explanation._model_args`

            assert isinstance(x, dict)
            explanation.set_value_dict('x', x)

            assert isinstance(edge_index, dict)
            explanation.set_value_dict('edge_index', edge_index)

            for key, arg in kwargs.items():  # Add remaining `kwargs`:
                if isinstance(arg, dict):
                    # Keyword arguments are likely named `{attr_name}_dict`
                    # while we only want to assign the `{attr_name}` to the
                    # `HeteroExplanation` object:
                    key = key[:-5] if key.endswith('_dict') else key
                    explanation.set_value_dict(key, arg)
                else:
                    explanation[key] = arg

        explanation.validate_masks()
        return explanation.threshold(self.threshold_config)

    def get_target(self, prediction: Tensor) -> Tensor:
        r"""Returns the target of the model from a given prediction.

        If the model mode is of type :obj:`"regression"`, the prediction is
        returned as it is.
        If the model mode is of type :obj:`"multiclass_classification"` or
        :obj:`"binary_classification"`, the prediction is returned as the
        predicted class label.
        """
        if self.model_config.mode == ModelMode.binary_classification:
            # TODO: Allow customization of the thresholds used below.
            if self.model_config.return_type == ModelReturnType.raw:
                return (prediction > 0).long().view(-1)
            if self.model_config.return_type == ModelReturnType.probs:
                return (prediction > 0.5).long().view(-1)
            assert False

        if self.model_config.mode == ModelMode.multiclass_classification:
            return prediction.argmax(dim=-1)

        return prediction
