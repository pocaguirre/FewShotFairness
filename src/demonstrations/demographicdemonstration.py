from .demonstration import Demonstration


class DemographicDemonstration(Demonstration):
    def __init__(self, shots: int = 16) -> None:
        """Base demonstration for demographic focused demonstrations

        :param shots: number of shots in demonstration, defaults to 16
        :type shots: int, optional
        """
        super().__init__(shots)

        self.type = "demographic"
