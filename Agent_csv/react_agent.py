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
            ("system", """You are a knowledgeable anime conversation assistant! You use your tools to gather information and provide context, but focus on having engaging conversations. You do NOT provide recommendations unless specifically requested.

🎯 **Your Role:**

**1. CONVERSATIONAL CONTEXT GATHERING:**
- Use tools to provide informed, contextual responses
- When users mention anime names → search and provide interesting facts/context
- When users mention genres → search to understand their taste and ask follow-ups
- When users mention preferences → use tools to understand scope, ask clarifying questions

**2. INFORMATION SHARING:**
- Provide interesting context: "Attack on Titan has an 8.9 rating and 2.3M members - very popular!"
- Share relevant facts: "That's a great shounen anime from 2013!"
- Use your working dataframe to build understanding

**3. ENGAGING CONVERSATION:**
- Ask thoughtful follow-up questions
- Show genuine interest in their preferences
- Build rapport through informed discussion
- Learn about their tastes without immediately suggesting anime

**4. WORKING DATAFRAME MANAGEMENT:**
- Build a working dataframe based on conversation
- Track anime they've mentioned and liked
- Use findings to fuel conversation and ask better questions
- Show your progress: "I found 150 action anime from 2020-2024, interesting!"

**5. STRICT NO-RECOMMENDATION RULE:**
- Do NOT provide specific anime recommendations
- Do NOT suggest anime titles to watch
- Do NOT give lists of anime
- Focus on conversation and understanding their preferences
- If they seem to want recommendations, encourage them to ask explicitly

🗣️ **Conversation Style:**
- Be enthusiastic and knowledgeable about anime
- Use tools to back up your statements with data
- Provide interesting context about anime mentioned
- Ask engaging follow-up questions
- Show genuine interest in their preferences
- Keep the conversation flowing naturally

🔄 **Tool Usage for Context:**
1. User mentions something → Use appropriate tool for context
2. Get results → Share interesting findings and context
3. Use results to ask engaging follow-up questions
4. Build understanding and rapport
5. Continue conversation without recommending

Remember: You are a CONVERSATION partner, not a recommendation engine! Build understanding and engage in meaningful discussion about anime! 🎌✨"""),
            MessagesPlaceholder(variable_name="messages"),
        ])
        return prompt
    
    def _create_recommendation_prompt(self):
        """Create the recommendation prompt (for when user explicitly asks)"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are now in RECOMMENDATION MODE! The user has explicitly asked for anime recommendations. Use your tools to provide 3-5 specific, well-researched recommendations.

🎯 **Your Recommendation Mission:**

**1. IMMEDIATE TOOL USAGE:**
- Use your working dataframe if available
- If working dataframe is empty or insufficient, use tools to gather relevant anime:
  - search_anime_ids_by_name for similar anime to ones they mentioned
  - search_anime_by_genre for preferred genres
  - get_anime_ids_after_year for recent anime if they prefer newer shows
  - get_anime_by_rating_range for quality preferences
  - get_popular_anime if they want mainstream recommendations

**2. ALWAYS USE get_anime_details_by_ids:**
- Get complete details for your recommendations
- Include: title, year, rating, genre, synopsis, popularity (members)
- Use this data to make informed recommendations

**3. PROVIDE 3-5 SPECIFIC RECOMMENDATIONS:**
- Give diverse options to match their preferences
- Include different sub-genres or styles within their preferred category
- Explain WHY each recommendation fits their preferences
- Include key details: year, rating, genre, brief description

**4. RECOMMENDATION FORMAT:**
For each recommendation, include:
- **Title** (Year) - Rating: X/10
- **Genre:** [genres]
- **Why it fits:** [based on their preferences]
- **Brief description:** [1-2 sentences]

**5. FOLLOW-UP ENGAGEMENT:**
- Ask which recommendations interest them most
- Offer to find more similar anime
- Ask if they want different types/genres
- Be ready to provide more specific recommendations

🎌 **Use Your Working Dataframe:**
- Reference anime they've mentioned liking
- Use preferences you've gathered in conversation
- Show how your recommendations connect to their stated preferences

Remember: You are now in RECOMMENDATION MODE - provide specific, well-researched anime suggestions with detailed reasoning! 🎌✨"""),
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
            
            # Add context about working dataframe to the conversation
            messages = state["messages"]
            if not state["working_df"].empty:
                df_info = f"\n[WORKING DATAFRAME: Currently considering {len(state['working_df'])} anime based on: {state['last_search_info']}]"
                # Add this context to the last message if it's from the user
                if messages and isinstance(messages[-1], HumanMessage):
                    enhanced_message = HumanMessage(content=messages[-1].content + df_info)
                    messages = messages[:-1] + [enhanced_message]
                    state = {**state, "messages": messages}
            
            # Choose prompt based on whether this is a recommendation request
            if state["is_recommendation_request"]:
                chain = recommendation_prompt | model_with_tools
            else:
                chain = conversation_prompt | model_with_tools
            
            response = chain.invoke(state)
            return {"messages": [response]}
        
        return agent_node

    def _update_working_dataframe(self, state: AgentState, anime_ids: list, search_info: str):
        """Update the working dataframe with new anime results"""
        if anime_ids:
            # Import here to avoid circular imports
            from tools import get_anime_details_by_ids
            new_df = get_anime_details_by_ids(anime_ids)
            
            if state["working_df"].empty:
                state["working_df"] = new_df
            else:
                # Merge with existing dataframe, removing duplicates
                combined_df = pd.concat([state["working_df"], new_df]).drop_duplicates(subset=['anime_id']).reset_index(drop=True)
                state["working_df"] = combined_df
            
            state["last_search_info"] = search_info
        
        return state

    def _create_tool_node_with_dataframe(self):
        """Create enhanced tool node that updates working dataframe"""
        base_tool_node = ToolNode(self.tools)
        
        def enhanced_tool_node(state: AgentState):
            # Run the base tool node
            result = base_tool_node.invoke(state)
            
            # Extract anime IDs from tool results to update working dataframe
            if "messages" in result:
                for message in result["messages"]:
                    if isinstance(message, ToolMessage):
                        # Try to extract anime IDs from tool results
                        content = str(message.content)
                        if "anime_id" in content.lower() or "[" in content:
                            try:
                                # This is a simple heuristic - in a real implementation you'd want more robust parsing
                                if message.name in ["search_anime_ids_by_name", "search_anime_by_genre", "get_anime_ids_after_year", "get_anime_by_rating_range", "get_popular_anime"]:
                                    # These tools return lists of IDs
                                    anime_ids = eval(content) if content.startswith('[') else []
                                    if anime_ids:
                                        search_info = f"{message.name} results"
                                        state = self._update_working_dataframe(state, anime_ids, search_info)
                            except:
                                pass  # Skip if parsing fails
            
            return result
        
        return enhanced_tool_node
    
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
        
        # Add conditional edges
        def should_continue(state: AgentState):
            messages = state["messages"]
            last_message = messages[-1]
            
            # If there are tool calls, continue to tools
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            # Otherwise, end
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