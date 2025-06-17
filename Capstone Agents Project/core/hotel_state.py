#hotel_state.py
from typing import TypedDict, Dict, Any

class HotelState(TypedDict):
    request: Dict[str, Any]
    booking: Dict[str, Any]
    housekeeping: Dict[str, Any]
    customer_service: Dict[str, Any]