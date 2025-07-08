import json
import os
from typing import Dict, Any

class UserPreferences:
    def __init__(self, preferences_file="user_preferences.json"):
        self.preferences_file = preferences_file
        self.default_preferences = {
            "answer_length": "medium",  # short, medium, long, detailed
            "answer_style": "conversational",  # conversational, formal, technical, simple
            "include_sources": True,
            "use_examples": True,
            "language": "english",
            "expertise_level": "intermediate"  # beginner, intermediate, advanced, expert
        }
        self.preferences = self.load_preferences()
    
    def load_preferences(self) -> Dict[str, Any]:
        """Load user preferences from file or create with defaults"""
        if os.path.exists(self.preferences_file):
            try:
                with open(self.preferences_file, 'r') as f:
                    saved_prefs = json.load(f)
                # Merge with defaults to ensure all keys exist
                preferences = self.default_preferences.copy()
                preferences.update(saved_prefs)
                return preferences
            except Exception as e:
                print(f"Error loading preferences: {e}")
                return self.default_preferences.copy()
        else:
            return self.default_preferences.copy()
    
    def save_preferences(self):
        """Save current preferences to file"""
        try:
            with open(self.preferences_file, 'w') as f:
                json.dump(self.preferences, f, indent=2)
            print("✅ Preferences saved!")
        except Exception as e:
            print(f"❌ Error saving preferences: {e}")
    
    def update_preference(self, key: str, value: Any):
        """Update a specific preference"""
        if key in self.default_preferences:
            self.preferences[key] = value
            self.save_preferences()
            print(f"Updated {key} to: {value}")
        else:
            print(f"Unknown preference: {key}")
    
    def get_system_prompt_addition(self) -> str:
        """Generate system prompt addition based on user preferences"""
        length_instructions = {
            "short": "Keep your answers brief and concise (1-2 sentences when possible).",
            "medium": "Provide balanced answers with key information (2-4 sentences).",
            "long": "Give comprehensive answers with good detail (multiple paragraphs).",
            "detailed": "Provide very detailed, thorough explanations with examples and context."
        }
        
        style_instructions = {
            "conversational": "Use a friendly, conversational tone.",
            "formal": "Use formal, professional language.",
            "technical": "Use precise technical language and terminology.",
            "simple": "Use simple language and avoid jargon."
        }
        
        level_instructions = {
            "beginner": "Explain concepts as if to someone new to the topic.",
            "intermediate": "Assume basic knowledge but explain advanced concepts.",
            "advanced": "Use technical language appropriate for experienced users.",
            "expert": "Provide expert-level analysis and insights."
        }
        
        prompt_additions = []
        
        # Answer length preference
        length = self.preferences.get("answer_length", "medium")
        prompt_additions.append(length_instructions.get(length, length_instructions["medium"]))
        
        # Answer style preference
        style = self.preferences.get("answer_style", "conversational")
        prompt_additions.append(style_instructions.get(style, style_instructions["conversational"]))
        
        # Expertise level
        level = self.preferences.get("expertise_level", "intermediate")
        prompt_additions.append(level_instructions.get(level, level_instructions["intermediate"]))
        
        # Sources preference
        if self.preferences.get("include_sources", True):
            prompt_additions.append("When possible, reference specific sources from the provided context.")
        
        # Examples preference
        if self.preferences.get("use_examples", True):
            prompt_additions.append("Include relevant examples when they help clarify your explanation.")
        
        return " ".join(prompt_additions)
    
    def show_current_preferences(self):
        """Display current user preferences"""
        print("\n📋 Current User Preferences:")
        print("-" * 40)
        for key, value in self.preferences.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        print("-" * 40)
    
    def interactive_setup(self):
        """Interactive preference setup"""
        print("\n🔧 User Preferences Setup")
        print("=" * 40)
        
        # Answer length
        print("\n📏 Answer Length Preference:")
        print("1. Short (brief, 1-2 sentences)")
        print("2. Medium (balanced, 2-4 sentences)")
        print("3. Long (comprehensive, multiple paragraphs)")
        print("4. Detailed (very thorough explanations)")
        
        choice = input("Choose (1-4) or press Enter for current: ").strip()
        if choice in ['1', '2', '3', '4']:
            lengths = ['short', 'medium', 'long', 'detailed']
            self.preferences['answer_length'] = lengths[int(choice) - 1]
        
        # Answer style
        print("\n🎨 Answer Style Preference:")
        print("1. Conversational (friendly, casual)")
        print("2. Formal (professional, structured)")
        print("3. Technical (precise, technical terms)")
        print("4. Simple (easy language, no jargon)")
        
        choice = input("Choose (1-4) or press Enter for current: ").strip()
        if choice in ['1', '2', '3', '4']:
            styles = ['conversational', 'formal', 'technical', 'simple']
            self.preferences['answer_style'] = styles[int(choice) - 1]
        
        # Expertise level
        print("\n🎓 Your Expertise Level:")
        print("1. Beginner (new to most topics)")
        print("2. Intermediate (some background knowledge)")
        print("3. Advanced (experienced user)")
        print("4. Expert (deep technical knowledge)")
        
        choice = input("Choose (1-4) or press Enter for current: ").strip()
        if choice in ['1', '2', '3', '4']:
            levels = ['beginner', 'intermediate', 'advanced', 'expert']
            self.preferences['expertise_level'] = levels[int(choice) - 1]
        
        self.save_preferences()
        print("\n✅ Preferences updated!")

# Example usage
if __name__ == "__main__":
    prefs = UserPreferences()
    prefs.interactive_setup()
    prefs.show_current_preferences()
    print("\nSystem prompt addition:")
    print(prefs.get_system_prompt_addition())
