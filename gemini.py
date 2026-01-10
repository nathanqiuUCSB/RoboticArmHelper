# gemini.py
from groq import Groq
import os
import json
from dotenv import load_dotenv

load_dotenv()

class NLPRobotPlanner:
    def __init__(self, api_key=None):
        """
        Initialize Gemini/OpenAI client
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)

    def parse_instruction(self, instruction):
        """
        Convert natural language instruction to structured JSON
        Example output:
        {
            "action": "pick_and_move",
            "color": "red",
            "direction": "right"
        }
        """
        prompt = f"""
        Convert this instruction into a JSON action plan:
        Instruction: "{instruction}"

        Output only valid JSON with these keys:
        - action: type of action (e.g., "pick_and_move", "pick", "move")
        - color: target object color (e.g., "red", "blue", "green")
        - direction: movement direction (e.g., "left", "right", "up", "down")
        
        Return ONLY the JSON object, no other text.
        """
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}]
            )
            result_text = response.choices[0].message.content
            
            # Remove markdown code blocks if present
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()
            
            action_plan = json.loads(result_text)
            
        except Exception as e:
            print(f"Error parsing response: {e}")
            action_plan = {"action": "unknown", "color": None, "direction": None}
        
        return action_plan

# Example usage
if __name__ == "__main__":
    planner = NLPRobotPlanner()
    instruction = "Pick up the red block and move it right"
    plan = planner.parse_instruction(instruction)
    print(plan)
