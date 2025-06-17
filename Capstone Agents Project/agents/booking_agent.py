import uuid
import random
from core.hotel_state import HotelState


def booking_agent(state: HotelState) -> HotelState:
    # extract request info
    customer_name = state["request"].get("customer_name", "Unknown Guest")
    room_type = state["request"].get("room_type", "Standard")

    # mock booking data
    room_number = random.randint(100, 999)
    booking_id = str(uuid.uuid4())
    booking_info = {
        "customer_name": customer_name,
        "room_type": room_type,
        "room_number": room_number,
        "booking_id": booking_id,
        "status": "booked"
    }

    print(f"✅ Booking Confirmed: {customer_name} in Room {room_number} ({room_type}) [ID: {booking_id}]")

    # return updated state dict, preserving previous keys
    return {**state, "booking": booking_info}
