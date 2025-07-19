import os
import pandas as pd
import re
from typing import Annotated, Sequence, TypedDict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from tools import tools

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Define the state of our agent with working dataframe
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    working_df: Optional[pd.DataFrame]  # Current filtered anime dataframe
    user_preferences: dict  # Track user preferences throughout conversation
    last_search_info: str  # Info about last search to show progress
    is_recommendation_request: bool  # Flag to indicate if user is asking for recommendations

class AnimeReactAgent:
    def __init__(self, model_name="gpt-4o-mini"):
        """
        Initialize the React Agent for anime recommendations using OpenAI GPT-4o mini.
        
        Args:
            model_name (str): OpenAI model name (default: gpt-4o-mini)
        """
        self.tools = tools
        self.model = self._setup_model(model_name)
        self.graph = self._create_graph()
        
    def _setup_model(self, model_name):
        """Setup the OpenAI language model"""
        return ChatOpenAI(model=model_name, temperature=0.3)  # Slightly higher temp for more natural conversation
    
    def _is_recommendation_request(self, user_input: str) -> bool:
        """
        Check if the user input is asking for recommendations
        
        Args:
            user_input (str): The user's message
            
        Returns:
            bool: True if asking for recommendations, False otherwise
        """
        user_input_lower = user_input.lower().strip()
        
        # Exact phrases that trigger recommendations
        recommendation_triggers = [
            "recommend now",
            "what do you recommend",
            "what should i watch",
            "show me some options",
            "give me suggestions",
            "give me recommendations",
            "what would you suggest",
            "any recommendations",
            "suggest something",
            "recommend something",
            "what do you think i should watch",
            "based on what i told you, what do you think",
            "what are your recommendations",
            "show me recommendations"
        ]
        
        # Check for exact matches
        for trigger in recommendation_triggers:
            if trigger in user_input_lower:
                return True
        
        # Check for question patterns that indicate recommendation requests
        recommendation_patterns = [
            r"\bwhat.*should.*watch\b",
            r"\bwhat.*recommend\b",
            r"\bany.*suggestions\b",
            r"\bany.*recommendations\b",
            r"\bshow.*me.*anime\b",
            r"\bgive.*me.*anime\b"
        ]
        
        for pattern in recommendation_patterns:
            if re.search(pattern, user_input_lower):
                return True
        
        return False
    
    def _create_conversation_prompt(self):
        """Create the conversation prompt (no recommendations)"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable anime conversation assistant! You MUST use your tools proactively after EVERY interaction to gather information and provide context. You do NOT provide recommendations unless specifically requested.

🎯 **MANDATORY TOOL USAGE - ALWAYS DO THIS:**

**1. IMMEDIATE TOOL USAGE AFTER EVERY MESSAGE:**
- ALWAYS use tools when users mention anime names, genres, years, or preferences
- NEVER respond without using tools when relevant information is mentioned
- Use multiple tools if needed to gather comprehensive context
- Build and maintain your working dataframe with every interaction

**2. SPECIFIC TOOL USAGE PATTERNS:**
- Anime name mentioned → MUST use search_anime_ids_by_name + get_anime_details_by_ids
- Genre mentioned → MUST use search_anime_by_genre + get_anime_details_by_ids
- Year/time period mentioned → MUST use get_anime_ids_after_year + get_anime_details_by_ids
- Popularity preferences → MUST use get_popular_anime + get_anime_details_by_ids
- Rating preferences → MUST use get_anime_by_rating_range + get_anime_details_by_ids
- General preferences → MUST use relevant search tools + get_anime_details_by_ids

**3. PROACTIVE INFORMATION GATHERING:**
- Use tools to provide rich context: "Attack on Titan has an 8.9 rating and 2.3M members - very popular!"
- Share detailed facts: "That's a great shounen anime from 2013 with 75 episodes!"
- Always get complete details about anime mentioned
- Use your working dataframe to build understanding

**4. WORKING DATAFRAME MANAGEMENT:**
- ALWAYS update your working dataframe with new findings
- Track anime they've mentioned and liked
- Use findings to fuel conversation and ask better questions
- Show your progress: "I found 150 action anime from 2020-2024, interesting!"

**5. ENGAGING CONVERSATION WITH DATA:**
- Ask thoughtful follow-up questions based on tool results
- Show genuine interest backed by data
- Build rapport through informed discussion
- Learn about their tastes using concrete information

**6. STRICT NO-RECOMMENDATION RULE:**
- Do NOT provide specific anime recommendations
- Do NOT suggest anime titles to watch
- Do NOT give lists of anime
- Focus on conversation and understanding their preferences
- If they seem to want recommendations, encourage them to ask explicitly

🔄 **MANDATORY WORKFLOW:**
1. User mentions anime/genre/preference → IMMEDIATELY use appropriate tools
2. Get comprehensive results → Share interesting findings and context
3. Use results to ask engaging follow-up questions
4. Build understanding and rapport with data
5. Continue conversation without recommending

⚠️ **CRITICAL RULE:** You MUST use tools proactively on every interaction where anime-related information is mentioned. Never respond without gathering relevant data first!

Remember: You are a CONVERSATION partner who ALWAYS uses tools to provide informed, data-backed responses! 🎌✨"""),
            MessagesPlaceholder(variable_name="messages"),
        ])
        return prompt
    
    def _create_recommendation_prompt(self):
        """Create the recommendation prompt (for when user explicitly asks)"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are now in RECOMMENDATION MODE! The user has explicitly asked for anime recommendations. You MUST use your tools proactively to provide 3-5 specific, well-researched recommendations.

🎯 **MANDATORY TOOL USAGE FOR RECOMMENDATIONS:**

**1. IMMEDIATE COMPREHENSIVE TOOL USAGE:**
- ALWAYS use tools to gather fresh, relevant anime data
- Use your working dataframe if available, but supplement with new searches
- MUST use multiple tools to ensure comprehensive recommendations:
  - search_anime_ids_by_name for similar anime to ones they mentioned
  - search_anime_by_genre for preferred genres
  - get_anime_ids_after_year for recent anime if they prefer newer shows
  - get_anime_by_rating_range for quality preferences
  - get_popular_anime if they want mainstream recommendations

**2. MANDATORY get_anime_details_by_ids USAGE:**
- ALWAYS use get_anime_details_by_ids for complete information
- Get details for ALL anime you plan to recommend
- Include: title, year, rating, genre, synopsis, popularity (members)
- Use this data to make informed, detailed recommendations

**3. PROACTIVE SEARCH STRATEGY:**
- Use 2-3 different search tools to gather diverse options
- Search based on their stated preferences and conversation history
- Cast a wide net, then select the best matches
- Always get complete details for potential recommendations

**4. PROVIDE 3-5 SPECIFIC RECOMMENDATIONS:**
- Give diverse options to match their preferences
- Include different sub-genres or styles within their preferred category
- Explain WHY each recommendation fits their preferences
- Include key details: year, rating, genre, brief description

**5. RECOMMENDATION FORMAT:**
For each recommendation, include:
- **Title** (Year) - Rating: X/10
- **Genre:** [genres]
- **Why it fits:** [based on their preferences]
- **Brief description:** [1-2 sentences]

**6. FOLLOW-UP ENGAGEMENT:**
- Ask which recommendations interest them most
- Offer to find more similar anime
- Ask if they want different types/genres
- Be ready to provide more specific recommendations

🎌 **Use Your Working Dataframe:**
- Reference anime they've mentioned liking
- Use preferences you've gathered in conversation
- Show how your recommendations connect to their stated preferences

⚠️ **CRITICAL RULE:** You MUST use tools to gather fresh data for recommendations. Never recommend based on memory alone!

Remember: You are now in RECOMMENDATION MODE - provide specific, well-researched anime suggestions with detailed reasoning backed by tool usage! 🎌✨"""),
            MessagesPlaceholder(variable_name="messages"),
        ])
        return prompt
    
    def _create_agent_node(self):
        """Create the reasoning/planning node with enhanced dataframe management"""
        conversation_prompt = self._create_conversation_prompt()
        recommendation_prompt = self._create_recommendation_prompt()
        model_with_tools = self.model.bind_tools(self.tools)
        
        def agent_node(state: AgentState):
            # Initialize state if not present
            if state.get("working_df") is None:
                state["working_df"] = pd.DataFrame()
            if state.get("user_preferences") is None:
                state["user_preferences"] = {}
            if state.get("last_search_info") is None:
                state["last_search_info"] = ""
            if state.get("is_recommendation_request") is None:
                state["is_recommendation_request"] = False
            
            # Get the user's latest message to analyze for proactive tool usage
            messages = state["messages"]
            user_message = ""
            if messages and isinstance(messages[-1], HumanMessage):
                user_message = messages[-1].content.lower()
            
            # Add context about working dataframe AND proactive tool usage hints
            enhanced_messages = messages.copy()
            if messages and isinstance(messages[-1], HumanMessage):
                df_info = ""
                if not state["working_df"].empty:
                    df_info = f"\n[WORKING DATAFRAME: Currently considering {len(state['working_df'])} anime based on: {state['last_search_info']}]"
                
                # Add proactive tool usage hints based on user message content
                tool_hints = self._generate_tool_usage_hints(user_message, state["is_recommendation_request"])
                
                enhanced_content = messages[-1].content + df_info + tool_hints
                enhanced_message = HumanMessage(content=enhanced_content)
                enhanced_messages = messages[:-1] + [enhanced_message]
            
            state = {**state, "messages": enhanced_messages}
            
            # Choose prompt based on whether this is a recommendation request
            if state["is_recommendation_request"]:
                chain = recommendation_prompt | model_with_tools
            else:
                chain = conversation_prompt | model_with_tools
            
            response = chain.invoke(state)
            return {"messages": [response]}
        
        return agent_node
    
    def _generate_tool_usage_hints(self, user_message: str, is_recommendation: bool):
        """Generate hints to force proactive tool usage based on user message"""
        hints = []
        
        # Check for anime names
        if any(keyword in user_message for keyword in ["attack on titan", "naruto", "one piece", "demon slayer", "death note", "fullmetal", "dragon ball", "bleach", "hunter x hunter", "my hero academia", "jujutsu kaisen", "chainsaw man", "spy x family", "cowboy bebop", "evangelion", "akira", "spirited away", "princess mononoke", "totoro", "kimetsu", "shingeki", "boku no hero", "jojo"]):
            hints.append("\n\n🔥 MANDATORY TOOL USAGE REQUIRED: User mentioned anime names - You MUST IMMEDIATELY use search_anime_ids_by_name + get_anime_details_by_ids before responding!")
        
        # Check for genres
        if any(keyword in user_message for keyword in ["action", "adventure", "comedy", "drama", "fantasy", "horror", "mystery", "romance", "sci-fi", "slice of life", "supernatural", "thriller", "shounen", "shoujo", "seinen", "josei", "mecha", "sports", "music", "school", "military", "psychological", "historical", "ecchi", "harem", "isekai", "magic", "vampire", "zombie", "idol", "game", "martial arts"]):
            hints.append("\n\n🔥 MANDATORY TOOL USAGE REQUIRED: User mentioned genres - You MUST IMMEDIATELY use search_anime_by_genre + get_anime_details_by_ids before responding!")
        
        # Check for time periods
        if any(keyword in user_message for keyword in ["recent", "new", "latest", "2020", "2021", "2022", "2023", "2024", "last year", "this year", "modern", "current", "ongoing", "airing"]):
            hints.append("\n\n🔥 MANDATORY TOOL USAGE REQUIRED: User mentioned time periods - You MUST IMMEDIATELY use get_anime_ids_after_year + get_anime_details_by_ids before responding!")
        
        # Check for popularity preferences
        if any(keyword in user_message for keyword in ["popular", "mainstream", "famous", "well-known", "trending", "top", "best", "classic", "must-watch", "everyone", "people", "fans"]):
            hints.append("\n\n🔥 MANDATORY TOOL USAGE REQUIRED: User mentioned popularity - You MUST IMMEDIATELY use get_popular_anime + get_anime_details_by_ids before responding!")
        
        # Check for quality preferences
        if any(keyword in user_message for keyword in ["good", "great", "excellent", "quality", "high rating", "rated", "score", "rating", "reviews", "acclaimed", "award", "critically"]):
            hints.append("\n\n🔥 MANDATORY TOOL USAGE REQUIRED: User mentioned quality - You MUST IMMEDIATELY use get_anime_by_rating_range + get_anime_details_by_ids before responding!")
        
        # General anime interest
        if any(keyword in user_message for keyword in ["anime", "manga", "watch", "watching", "seen", "like", "love", "enjoy", "favorite", "prefer", "interested", "looking for", "want", "need", "suggest", "similar", "type", "style", "series", "show", "episode"]):
            if not hints:  # Only add if no specific hints were found
                hints.append("\n\n🔥 MANDATORY TOOL USAGE REQUIRED: User mentioned anime-related terms - You MUST IMMEDIATELY use appropriate search tools + get_anime_details_by_ids before responding!")
        
        # Force comprehensive tool usage for recommendations
        if is_recommendation:
            hints.append("\n\n🔥 RECOMMENDATION MODE - You MUST IMMEDIATELY use multiple search tools + get_anime_details_by_ids for ALL recommendations!")
        
        return "".join(hints)

    def _update_working_dataframe(self, state: AgentState, anime_ids: list, search_info: str):
        """Update the working dataframe with new anime results"""
        if anime_ids:
            try:
                # Import here to avoid circular imports
                from tools import get_anime_details_by_ids
                
                # Use the correct tool invocation format
                new_df = get_anime_details_by_ids.invoke({"anime_ids": anime_ids})
                
                if state["working_df"].empty:
                    state["working_df"] = new_df
                else:
                    # Merge with existing dataframe, removing duplicates
                    combined_df = pd.concat([state["working_df"], new_df]).drop_duplicates(subset=['anime_id']).reset_index(drop=True)
                    state["working_df"] = combined_df
                
                state["last_search_info"] = search_info
                
            except Exception as e:
                print(f"Error updating working dataframe: {e}")
                # Return state unchanged if error occurs
                return state
        
        return state

    def _create_tool_node_with_dataframe(self):
        """Create enhanced tool node that updates working dataframe"""
        base_tool_node = ToolNode(self.tools)
        
        def enhanced_tool_node(state: AgentState):
            # Run the base tool node
            result = base_tool_node.invoke(state)
            
            # Track tool usage and extract anime IDs
            anime_ids_found = []
            search_info_parts = []
            
            if "messages" in result:
                for message in result["messages"]:
                    if isinstance(message, ToolMessage):
                        content = str(message.content)
                        tool_name = getattr(message, 'name', 'unknown')
                        
                        # Handle search tools that return anime IDs
                        if tool_name in ["search_anime_ids_by_name", "search_anime_by_genre", "get_anime_ids_after_year", "get_anime_by_rating_range", "get_popular_anime"]:
                            try:
                                # These tools return lists of IDs
                                if content.startswith('[') and content.endswith(']'):
                                    anime_ids = eval(content)
                                    if isinstance(anime_ids, list) and anime_ids:
                                        anime_ids_found.extend(anime_ids)
                                        search_info_parts.append(f"{tool_name}")
                            except:
                                pass  # Skip if parsing fails
                        
                        # Handle get_anime_details_by_ids - this updates working dataframe differently
                        elif tool_name == "get_anime_details_by_ids":
                            # This tool returns a dataframe, we don't need to extract IDs
                            # The dataframe management happens in the agent logic
                            pass
                        
                        # Log tool usage for debugging
                        print(f"🔧 Tool used: {tool_name}")
                        if anime_ids_found:
                            print(f"📊 Found {len(anime_ids_found)} anime IDs")
            
            # Update working dataframe with all collected anime IDs from search tools
            if anime_ids_found:
                search_info = " + ".join(search_info_parts)
                state = self._update_working_dataframe(state, anime_ids_found, search_info)
                print(f"📋 Updated working dataframe with {len(anime_ids_found)} entries from: {search_info}")
            
            return result
        
        return enhanced_tool_node
    
    def _should_force_tool_usage(self, state: AgentState):
        """Check if we should force tool usage based on user message content"""
        messages = state["messages"]
        if not messages or len(messages) < 2:
            return False
        
        # Get the user's message (skip system override messages)
        user_message = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) and "[SYSTEM OVERRIDE:" not in msg.content:
                user_message = msg.content.lower()
                break
        
        # Don't force if no user message found
        if not user_message:
            return False
        
        # Get the agent's response
        last_message = messages[-1]
        has_tool_calls = hasattr(last_message, 'tool_calls') and last_message.tool_calls
        
        # If agent already used tools, don't force
        if has_tool_calls:
            return False
        
        # Check if we're already in a forced retry loop (prevent infinite loops)
        if any("[SYSTEM OVERRIDE:" in msg.content for msg in messages if isinstance(msg, HumanMessage)):
            print("🔄 Already in forced retry mode, stopping to prevent infinite loop")
            return False
        
        # Check if user mentioned anime-related terms that should trigger tool usage
        anime_keywords = [
            # Anime names
            "attack on titan", "naruto", "one piece", "demon slayer", "death note", 
            "fullmetal", "dragon ball", "bleach", "hunter x hunter", "my hero academia",
            "jujutsu kaisen", "chainsaw man", "spy x family", "cowboy bebop", "evangelion",
            
            # Genres
            "action", "adventure", "comedy", "drama", "fantasy", "horror", "mystery", 
            "romance", "sci-fi", "slice of life", "supernatural", "thriller", "shounen",
            "shoujo", "seinen", "josei", "mecha", "sports", "isekai", "magic",
            
            # General anime terms
            "anime", "manga", "watch", "watching", "seen", "like", "love", "enjoy",
            "favorite", "prefer", "interested", "looking for", "want", "similar",
            "recent", "new", "latest", "popular", "mainstream", "good", "great",
            "quality", "rating", "recommend", "suggest", "type", "style", "series"
        ]
        
        # Check if any anime keywords are mentioned
        for keyword in anime_keywords:
            if keyword in user_message:
                print(f"🚨 FORCING TOOL USAGE: User mentioned '{keyword}' but agent didn't use tools")
                return True
        
        return False

    def _create_graph(self):
        """Create the enhanced LangGraph workflow"""
        # Create nodes
        agent_node = self._create_agent_node()
        tool_node = self._create_tool_node_with_dataframe()
        
        # Define the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges with simplified tool checking
        def should_continue(state: AgentState):
            messages = state["messages"]
            last_message = messages[-1]
            
            # Debug output
            print(f"🔍 Checking should_continue: last_message type = {type(last_message)}")
            if hasattr(last_message, 'tool_calls'):
                print(f"🔍 Tool calls present: {bool(last_message.tool_calls)}")
                if last_message.tool_calls:
                    print(f"🔍 Tool calls: {[tc.get('name', 'unknown') for tc in last_message.tool_calls]}")
            
            # If there are tool calls, continue to tools
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                print("✅ Continuing to tools")
                return "tools"
            
            # Otherwise, end
            print("🔚 Ending conversation")
            return END
        
        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")
        
        # Compile the graph
        return workflow.compile()
    
    def chat(self, message: str, state: Optional[AgentState] = None):
        """
        Chat with the agent, maintaining conversation state
        
        Args:
            message (str): User's message/question
            state (Optional[AgentState]): Previous conversation state
            
        Returns:
            tuple: (response_string, updated_state)
        """
        try:
            # Check if this is a recommendation request
            is_recommendation = self._is_recommendation_request(message)
            
            # Create or update state
            if state is None:
                initial_state = {
                    "messages": [HumanMessage(content=message)],
                    "working_df": pd.DataFrame(),
                    "user_preferences": {},
                    "last_search_info": "",
                    "is_recommendation_request": is_recommendation
                }
            else:
                initial_state = {
                    **state,
                    "messages": state["messages"] + [HumanMessage(content=message)],
                    "is_recommendation_request": is_recommendation
                }
            
            # Run the graph
            result = self.graph.invoke(initial_state)
            
            # Get the final message
            final_message = result["messages"][-1]
            return final_message.content, result
            
        except Exception as e:
            return f"Sorry, I encountered an error: {e}", state

    def simple_chat(self, message: str):
        """
        Simple chat interface that doesn't maintain state between calls
        (for backward compatibility)
        """
        response, _ = self.chat(message)
        return response
    
    def stream_chat(self, message: str, state: Optional[AgentState] = None):
        """
        Stream chat responses for real-time interaction
        
        Args:
            message (str): User's message/question
            state (Optional[AgentState]): Previous conversation state
            
        Yields:
            dict: Streaming updates from the agent
        """
        try:
            # Check if this is a recommendation request
            is_recommendation = self._is_recommendation_request(message)
            
            if state is None:
                initial_state = {
                    "messages": [HumanMessage(content=message)],
                    "working_df": pd.DataFrame(),
                    "user_preferences": {},
                    "last_search_info": "",
                    "is_recommendation_request": is_recommendation
                }
            else:
                initial_state = {
                    **state,
                    "messages": state["messages"] + [HumanMessage(content=message)],
                    "is_recommendation_request": is_recommendation
                }
            
            for update in self.graph.stream(initial_state):
                yield update
                
        except Exception as e:
            yield {"error": f"Error during streaming: {e}"}

def create_agent(model_name="gpt-4o-mini"):
    """
    Factory function to create an anime recommendation agent using OpenAI GPT-4o mini
    
    Args:
        model_name (str): OpenAI model name (default: gpt-4o-mini)
        
    Returns:
        AnimeReactAgent: Configured agent instance
    """
    return AnimeReactAgent(model_name)

# Example usage and testing
if __name__ == "__main__":
    print("🎌 Enhanced Anime Recommendation Agent Setup (OpenAI GPT-4o mini)")
    print("=" * 70)
    
    try:
        # Create agent
        print("Creating enhanced OpenAI recommendation agent with GPT-4o mini...")
        agent = create_agent()
        print("✅ Agent created successfully!")
        
        # Test conversation flow with state management
        print("\n🧪 Running enhanced test conversation...")
        
        conversation_state = None
        test_conversations = [
            "I really liked Attack on Titan",
            "I want something recent and popular",
            "What do you recommend?"
        ]
        
        for i, message in enumerate(test_conversations, 1):
            print(f"\n--- User: {message} ---")
            try:
                response, conversation_state = agent.chat(message, conversation_state)
                print(f"Agent: {response}")
                
                # Show working dataframe info
                if conversation_state and not conversation_state["working_df"].empty:
                    print(f"[Working with {len(conversation_state['working_df'])} anime: {conversation_state['last_search_info']}]")
                    
            except Exception as e:
                print(f"Error: {e}")
                
    except Exception as e:
        print(f"❌ Setup error: {e}")
        print("\n💡 Make sure to set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-openai-key'")
        print("  Ensure all dependencies are installed: pip install -r requirements.txt") 