# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================


class Example:
    """
    Examples are input/output pairs that represent inputs to a function and then
    expected output. They can be used in both training and evaluation of models.

    These can be inputs/outputs for a model or for a chain. Both types of
    examples serve a different purpose. Examples for a model can be used to
    finetune a model. Examples for a chain can be used to evaluate the
    end-to-end chain, or maybe even train a model to replace that whole chain.
    """
    def __init__(self, id: str, input: str, output: str) -> None:
        self._id = id
        self._input = input
        self._output = output

    @property
    def id(self) -> str:
        """Returns the unique ID of the example."""
        return self._id

    @property
    def input(self) -> str:
        """Returns the input of the example."""
        return self._input

    @property
    def output(self) -> str:
        """Returns the intended output of the example."""
        return self._output

    def dict(self) -> dict:
        return {
            "id": self._id,
            "input": self._input,
            "output": self._output,
        }

    def __str__(self) -> str:
        return str(self.dict())

    def __repr__(self) -> str:
        return str(self.dict())
