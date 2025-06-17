from typing import Dict, Any

class HotelState:
    """
    Shared state object containing:
    - request: input data (customer info, room type)
    - booking: booking status data
    - housekeeping: housekeeping status data
    - customer_service: customer service interactions
    """
    def __init__(self, request: Dict[str, Any]):
        self.request = request
        self.booking: Dict[str, Any] = {}
        self.housekeeping: Dict[str, Any] = {}
        self.customer_service: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert the state to a dictionary for serialization/logging."""
        return {
            "request": self.request,
            "booking": self.booking,
            "housekeeping": self.housekeeping,
            "customer_service": self.customer_service
        }