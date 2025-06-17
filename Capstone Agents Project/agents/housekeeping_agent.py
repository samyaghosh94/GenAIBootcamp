import random
from core.hotel_state import HotelState

def housekeeping_agent(state: HotelState) -> HotelState:
    booking = state.get("booking", {})

    if not booking:
        print("⚠️ No booking found. Cannot perform housekeeping.")
        return state

    room_type = booking.get("room_type", "Standard")
    room_number = booking.get("room_number", "Unknown")

    housekeepers = ["Jamie", "Riley", "Morgan", "Alex", "Taylor"]
    assigned_housekeeper = random.choice(housekeepers)

    housekeeping_info = {
        "status": "cleaning scheduled",
        "assigned_housekeeper": assigned_housekeeper,
        "room_number": room_number,
        "room_type": room_type
    }

    print(f"🧹 Housekeeping: Room {room_number} ({room_type}) is now 'Ready' [Assigned to: {assigned_housekeeper}]")

    return {**state, "housekeeping": housekeeping_info}
