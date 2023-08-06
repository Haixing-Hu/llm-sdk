# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Example:
    """
    Examples are input/output pairs that represent inputs to a function and then
    expected output. They can be used in both training and evaluation of models.

    These can be inputs/outputs for a model or for a chain. Both types of
    examples serve a different purpose. Examples for a model can be used to
    finetune a model. Examples for a chain can be used to evaluate the
    end-to-end chain, or maybe even train a model to replace that whole chain.
    """

    input: str
    """The input of the example"""

    output: str
    """The output of the example"""

    id: str = None
    """The ID of the example."""

    score: Optional[float] = None
    """The score of this example relevant to the query."""

    def __eq__(self, other):
        """
        Tests whether this object is equal to another object.

        Two examples are equal if and only if all their fields except the score
        field are equal.

        :param other: the other object.
        :return; true if this object is equal to the other object; false otherwise.
        """
        if isinstance(other, Example):
            return (self.id == other.id
                    and self.input == other.input
                    and self.output == other.output)
        else:
            return False
