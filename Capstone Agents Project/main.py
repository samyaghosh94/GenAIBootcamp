from core.hotel_state import HotelState
from agents.booking_agent import booking_agent
from agents.housekeeping_agent import housekeeping_agent
from agents.customer_service_agent import customer_service_agent

def run_hotel_workflow():
    # Step 1: Construct initial request
    initial_request = {
        "inquiry": "Do you have late checkout?",  # match key to customer_service_agent expected 'inquiry'
        "customer_name": "Alice Johnson",
        "room_type": "Deluxe",
    }

    # Step 2: Create shared state
    state = HotelState(initial_request)
    state.booking = {"customer_name": initial_request["customer_name"]}  # ensure booking info is present

    print("\n🏨 Hotel Management Workflow Starting...\n")

    # Step 3: Booking
    state = booking_agent(state)

    # Step 4: Housekeeping
    state = housekeeping_agent(state)

    # Step 5: Customer Service (RAG-enabled)
    state = customer_service_agent(state)

    print("\n✅ Workflow Complete. Final State:\n")
    print(state.to_dict())

if __name__ == "__main__":
    run_hotel_workflow()
