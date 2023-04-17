from typing import List, Set

from .demonstration import Demonstration


class DemographicDemonstration(Demonstration):
    def __init__(self, shots: int = 16) -> None:
        super().__init__(shots)

        self.type = "demographic"

    def filter_demographics(
        self, demographics: List[str], overall_demographics: Set[str]
    ) -> str:

        set_of_demographics = set(demographics)

        intersection = set_of_demographics.intersection(overall_demographics)

        if len(intersection) == 0:
            return ""

        else:
            return list(intersection)[0]
