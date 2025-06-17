import random
from core.hotel_state import HotelState

def housekeeping_agent(state: HotelState) -> HotelState:
    """
    Simulates housekeeping readiness after a room booking.

    Steps:
    - Retrieves room type and room number from booking.
    - Assigns a random housekeeper.
    - Updates state.housekeeping with status and staff info.
    - Prints a housekeeping confirmation.
    """
    booking = state.booking

    if not booking:
        print("⚠️ No booking found. Cannot perform housekeeping.")
        return state

    room_type = booking.get("room_type", "Standard")
    room_number = booking.get("room_number", "Unknown")

    # Mock housekeeper names
    housekeepers = ["Jamie", "Riley", "Morgan", "Alex", "Taylor"]
    assigned_housekeeper = random.choice(housekeepers)

    # Update housekeeping info
    housekeeping_info = {
        "status": "Ready",
        "room_number": room_number,
        "room_type": room_type,
        "assigned_to": assigned_housekeeper
    }

    state.housekeeping = housekeeping_info

    # Print confirmation
    print(f"🧹 Housekeeping: Room {room_number} ({room_type}) is now 'Ready' [Assigned to: {assigned_housekeeper}]")

    return state
