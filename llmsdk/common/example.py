# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from dataclasses import dataclass


@dataclass
class Example:
    """
    Examples are input/output pairs that represent inputs to a function and then
    expected output. They can be used in both training and evaluation of models.

    These can be inputs/outputs for a model or for a chain. Both types of
    examples serve a different purpose. Examples for a model can be used to
    finetune a model. Examples for a chain can be used to evaluate the
    end-to-end chain, or maybe even train a model to replace that whole chain.
    """

    id: str
    """The ID of the example."""

    input: str
    """The input of the example"""

    output: str
    """The output of the example"""
