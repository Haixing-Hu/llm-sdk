# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from dataclasses import dataclass
from typing import Dict

from .role import Role, ROLE_NAMES_MAP


@dataclass(frozen=True)
class Message:
    """
    The data structure represents chatting messages.
    """

    role: Role
    """
    The role of the speaker.
    """

    content: str
    """
    The content of the message.
    """

    name: str = None
    """
    The optional name of the speaker.
    """

    def to_dict(self,
                role_names_map: Dict[Role, str] = ROLE_NAMES_MAP) -> Dict[str, str]:
        """
        Converts this message to a dictionary.

        :param role_names_map: the map which maps a enumerator of Role into its
          name.
        :return: the dictionary converted from this message.
        """
        role_name = role_names_map[self.role]
        result = {
            "role": role_name,
            "content": self.content,
        }
        if self.name is not None:
            result["name"] = self.name
        return result
