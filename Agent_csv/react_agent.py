import os
from typing import Annotated, Sequence, TypedDict
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

# Define the state of our agent
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

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
    
    def _create_prompt(self):
        """Create the anime recommendation agent prompt"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a friendly anime recommendation assistant! Your goal is to have natural conversations with users to understand their anime preferences, then use your tools to find and recommend anime they'll love.

🎯 **Your Approach:**
1. **CHAT FIRST**: Start conversations by getting to know the user's preferences
   - Ask about their favorite anime, genres, or what they're in the mood for
   - Learn about anime they've watched and liked/disliked
   - Understand their preferences (action, romance, comedy, etc.)
   - Ask about specific years, ratings, or popularity preferences

2. **LISTEN & UNDERSTAND**: Pay attention to what they tell you about:
   - Anime they've enjoyed
   - Genres they prefer or want to avoid
   - Whether they want recent anime or don't mind older ones
   - If they want popular mainstream anime or hidden gems
   - Mood they're in (want something light, serious, exciting, etc.)

🚨 **CRITICAL: SPECIAL FORCE COMMAND "RECOMMEND NOW"**
When the user says "RECOMMEND NOW" (in any capitalization), you MUST:
1. IMMEDIATELY stop asking questions
2. USE YOUR TOOLS RIGHT NOW to find recommendations
3. Base recommendations on ANY information shared so far in the conversation
4. If they mentioned specific anime they liked: use search_anime_ids_by_name to find similar
5. If they mentioned wanting recent anime: use get_anime_ids_after_year 
6. If you have minimal info: search for popular recent anime (use get_anime_ids_after_year with 2020)
7. ALWAYS use get_anime_details_by_ids to get full details for your recommendations
8. Give 3-5 specific anime recommendations with details
9. DO NOT ask for more information - just recommend based on what you know

This is a FORCE COMMAND that bypasses all conversation and triggers immediate recommendations.

3. **RECOGNIZE OTHER RECOMMENDATION TRIGGERS**: Make recommendations when users say:
   - "What do you recommend?"
   - "Show me some options"
   - "Give me recommendations"
   - "What should I watch?"
   - "I'm ready for suggestions"
   - "Based on what I told you, what do you think?"
   - Or after you've gathered enough preference information (2-3 exchanges)

4. **USE TOOLS STRATEGICALLY**: When ready to recommend, use your tools:
   - `get_anime_ids_after_year`: If they mention wanting recent anime or anime from specific years
   - `search_anime_ids_by_name`: If they mention specific anime names or want similar shows
   - `get_anime_details_by_ids`: To get full details and make informed recommendations

5. **RECOMMEND THOUGHTFULLY**: 
   - Explain WHY you're recommending specific anime based on their preferences
   - Include details like year, rating, genre, and brief description
   - Suggest 3-5 anime with variety to give them options
   - Ask if they want more recommendations or different types
   - Offer to explain more about any specific recommendation

6. **CONTINUE THE CONVERSATION**: After recommendations:
   - Ask what they think of the suggestions
   - Offer to find more similar anime
   - Ask if they want different genres or types
   - Be ready to refine based on their feedback

🗣️ **Conversation Style:**
- Be friendly, enthusiastic, and conversational
- Ask engaging questions about their anime preferences
- Show genuine interest in their tastes
- Use emojis and casual language to feel approachable
- Don't jump straight to using tools - chat first!
- When you have enough info or they ask, transition smoothly to recommendations
- EXCEPT when they say "RECOMMEND NOW" - then immediately use tools and recommend!

📋 **Available Tools:**
- get_anime_ids_after_year: Get anime that aired after a specific year
- search_anime_ids_by_name: Search for anime by name (partial matching)
- get_anime_details_by_ids: Get complete details about specific anime

🎌 **User Signals for Recommendations:**
- **FORCE KEYWORD**: "RECOMMEND NOW" = IMMEDIATE tool usage and recommendations with ALL available info (NO MORE QUESTIONS!)
- Direct requests: "recommend", "suggest", "what should I watch"
- Completion signals: "that's all", "I think you have enough info"
- Question format: "what do you think?", "any ideas?"
- After sharing 2-3 preferences, proactively offer to recommend

Remember: "RECOMMEND NOW" means STOP TALKING, START USING TOOLS, GIVE RECOMMENDATIONS NOW! 🎌✨"""),
            MessagesPlaceholder(variable_name="messages"),
        ])
        return prompt
    
    def _create_agent_node(self):
        """Create the reasoning/planning node"""
        prompt = self._create_prompt()
        model_with_tools = self.model.bind_tools(self.tools)
        
        def agent_node(state: AgentState):
            chain = prompt | model_with_tools
            response = chain.invoke(state)
            return {"messages": [response]}
        
        return agent_node
    
    def _create_graph(self):
        """Create the LangGraph workflow"""
        # Create nodes
        agent_node = self._create_agent_node()
        tool_node = ToolNode(self.tools)
        
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
    
    def chat(self, message: str):
        """
        Chat with the agent
        
        Args:
            message (str): User's message/question
            
        Returns:
            str: Agent's response
        """
        try:
            # Create initial state
            initial_state = {"messages": [HumanMessage(content=message)]}
            
            # Run the graph
            result = self.graph.invoke(initial_state)
            
            # Get the final message
            final_message = result["messages"][-1]
            return final_message.content
            
        except Exception as e:
            return f"Sorry, I encountered an error: {e}"
    
    def stream_chat(self, message: str):
        """
        Stream chat responses for real-time interaction
        
        Args:
            message (str): User's message/question
            
        Yields:
            dict: Streaming updates from the agent
        """
        try:
            initial_state = {"messages": [HumanMessage(content=message)]}
            
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
    print("🎌 Anime Recommendation Agent Setup (OpenAI GPT-4o mini)")
    print("=" * 65)
    
    try:
        # Create agent
        print("Creating OpenAI recommendation agent with GPT-4o mini...")
        agent = create_agent()
        print("✅ Agent created successfully!")
        
        # Test conversation flow
        test_conversations = [
            "Hi! I'm looking for some anime recommendations",
            "I really liked Attack on Titan and Death Note",
            "I want something recent and popular"
        ]
        
        print("\n🧪 Running test conversation...")
        for i, message in enumerate(test_conversations, 1):
            print(f"\n--- User: {message} ---")
            try:
                response = agent.chat(message)
                print(f"Agent: {response}")
            except Exception as e:
                print(f"Error: {e}")
                
    except Exception as e:
        print(f"❌ Setup error: {e}")
        print("\n💡 Make sure to set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-key-here'")
        print("   Get your key at: https://platform.openai.com/api-keys") 