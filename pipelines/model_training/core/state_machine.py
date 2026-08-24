from enum import Enum
from typing import Any


class State(Enum):
    PRESENTATION = "presentation"
    HISTORY_REVEALED = "history_revealed"
    ASSESSMENT = "assessment"
    ESCALATION = "escalation"

class PersonaStateMachine:
    def __init__(self, session_id: str, persona_definition: dict[str, Any]):
        self.session_id = session_id
        self.persona_definition = persona_definition
        self.current_state = State.PRESENTATION
        self.variables = {
            "pain_level": persona_definition.get("clinical_profile", {}).get("symptoms", [{}])[0].get("severity", 5),
            "anxiety_level": persona_definition.get("emotional_state", {}).get("volatility", 5),
            "disclosed_symptoms": [],
            "hidden_symptoms": [s["name"] for s in persona_definition.get("clinical_profile", {}).get("symptoms", [])]
        }
        self.history = []
        self.turn_count = 0
        self.neglect_count = 0

    def transition(self, action: str, metadata: dict[str, Any] | None = None) -> State:
        self.turn_count += 1
        action_entry = {"turn": self.turn_count, "action": action, "state_before": self.current_state.value}


        if self.current_state == State.PRESENTATION:
            if action == "ask_history":
                self.current_state = State.HISTORY_REVEALED
                # Disclose some hidden symptoms if relevant
                if "medical_history" in self.persona_definition.get("clinical_profile", {}):
                    self.variables["disclosed_symptoms"].append("medical_history")
            elif action == "neglect":
                self.neglect_count += 1
                if self.neglect_count >= 3:
                    self.current_state = State.ESCALATION
                    self.variables["pain_level"] = min(10, self.variables["pain_level"] + 2)
                    self.variables["anxiety_level"] = min(10, self.variables["anxiety_level"] + 2)
            else:
                self.neglect_count = 0 # Reset neglect if something else is done?
                # Or keep it if pain isn't addressed? The spec says "3 turns without addressing pain"

        elif self.current_state == State.HISTORY_REVEALED:
            if action == "perform_intervention":
                self.current_state = State.ASSESSMENT
            elif action == "neglect":
                # Maybe history revealed doesn't easily escalate or has different logic
                pass

        elif self.current_state == State.ESCALATION and (action in {"address_pain", "soothe"}):
            self.current_state = State.PRESENTATION # De-escalate
            self.neglect_count = 0
            self.variables["anxiety_level"] = max(1, self.variables["anxiety_level"] - 1)

        action_entry["state_after"] = self.current_state.value
        self.history.append(action_entry)

        return self.current_state

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "current_state": self.current_state.value,
            "variables": self.variables,
            "history": self.history,
            "turn_count": self.turn_count,
            "neglect_count": self.neglect_count
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], persona_definition: dict[str, Any]) -> "PersonaStateMachine":
        instance = cls(data["session_id"], persona_definition)
        instance.current_state = State(data["current_state"])
        instance.variables = data["variables"]
        instance.history = data["history"]
        instance.turn_count = data.get("turn_count", len(data["history"]))
        instance.neglect_count = data.get("neglect_count", 0)
        return instance
