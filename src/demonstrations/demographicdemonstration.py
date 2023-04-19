from typing import List, Set

from .demonstration import Demonstration


class DemographicDemonstration(Demonstration):
    def __init__(self, shots: int = 16) -> None:
        super().__init__(shots)

        self.type = "demographic"


