# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from typing import Union, TypeAlias

from .message import MessageList


Prompt: TypeAlias = Union[str, MessageList]
"""
The type of prompts. 

A “prompt” refers to what is passed to the underlying model. The main 
abstractions have for prompt in this library so for all deal with text data. 
For other data types (images, audio) we are working on adding abstractions but 
do not yet have them.

Different models may expect different data formats. Where possible, we want to 
allow for the same prompt to be used in different model types. For that reason, 
we have a concept of a Prompt. 

Currently a prompt is either a string or a list of chatting messages.
"""
