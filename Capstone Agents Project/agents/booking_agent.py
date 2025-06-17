import uuid
import random
from core.hotel_state import HotelState

def booking_agent(state: HotelState) -> HotelState:
    """
    Simulates room booking.

    Steps:
    - Extracts customer and room type from the request.
    - Generates a unique booking ID.
    - Mocks room assignment and sets status to 'Booked'.
    - Updates state.booking with booking details.
    - Prints a confirmation message.
    """
    customer_name = state.request.get("customer_name", "Unknown Guest")
    room_type = state.request.get("room_type", "Standard")

    # Mock room assignment (e.g., pick a random room number in a range)
    room_number = random.randint(100, 999)

    # Generate a unique booking ID
    booking_id = str(uuid.uuid4())

    # Mock booking details
    booking_info = {
        "booking_id": booking_id,
        "customer_name": customer_name,
        "room_type": room_type,
        "room_number": room_number,
        "status": "Booked"
    }

    # Update the shared state
    state.booking = booking_info

    # Print confirmation
    print(f"✅ Booking Confirmed: {customer_name} in Room {room_number} ({room_type}) [ID: {booking_id}]")

    return state
