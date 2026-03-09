from abc import ABC, abstractmethod
from typing import Union, Dict, Any

class BaseFetcher(ABC):
    def __init__(self, source_name: str, config: Dict[str, Any] = None):
        self.source_name = source_name
        self.config = config or {}

    @abstractmethod
    def fetch_data(self, start_time: str, end_time: str, bbox: tuple = None) -> Any:
        pass

    @abstractmethod
    def validate_data(self, data: Any) -> bool:
        pass

    def process(self, start_time: str, end_time: str, bbox: tuple = None) -> Any:
        print(f"[{self.source_name}] Starting fetch for {start_time}...")
        data = self.fetch_data(start_time, end_time, bbox)
        
        if data is None:
            print(f"[{self.source_name}] No data found for given parameters.")
            return None
            
        if self.validate_data(data):
            return data
        else:
            print(f"[{self.source_name}] Data validation failed.")
            return None