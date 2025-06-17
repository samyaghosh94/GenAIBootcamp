from core.hotel_state import HotelState
from agents.booking_agent import booking_agent
from agents.housekeeping_agent import housekeeping_agent
from agents.customer_service_agent import customer_service_agent
from langgraph.graph import StateGraph


def build_graph() -> StateGraph:
    graph = StateGraph(HotelState)

    # Register nodes
    graph.add_node("booking_agent", booking_agent)
    graph.add_node("housekeeping_agent", housekeeping_agent)
    graph.add_node("customer_service_agent", customer_service_agent)

    graph.set_entry_point("booking_agent")

    graph.add_edge("booking_agent", "housekeeping_agent")
    graph.add_edge("housekeeping_agent", "customer_service_agent")

    graph.set_finish_point("customer_service_agent")

    return graph.compile()  # returns a compiled graph object, but no need to type explicitly


if __name__ == "__main__":
    initial_request = {
        "customer_name": "Bob Smith",
        "room_type": "Suite",
        "inquiry": "Do you offer late checkout?"
    }

    initial_state = {
        "request": initial_request,
        "booking": {},
        "housekeeping": {},
        "customer_service": {}
    }

    graph = build_graph()

    print(graph.get_graph().draw_mermaid())

    print("\n🔁 Running LangGraph Workflow...\n")

    final_state = graph.invoke(initial_state)

    print("\n--- Final State ---")
    print("Booking Info:", final_state["booking"])
    print("Housekeeping Info:", final_state["housekeeping"])
    print("Customer Service Info:", final_state["customer_service"])
