from abc import ABC, abstractmethod

class BasePipeline(ABC):
    """Abstract Base Class for all Pipelines"""

    @abstractmethod
    def _execute(self):
        """The pipeline logic"""
        pass

    def run(self):
        """Runs the pipeline"""
        return self._execute()