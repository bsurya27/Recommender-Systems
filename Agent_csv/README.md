# Anime Recommendation Assistant 🎌🤖

A conversational anime recommendation agent powered by **OpenAI GPT-4o mini** and LangGraph. The agent chats with you to understand your preferences, then uses a comprehensive anime database to recommend shows you'll love!

## 🚀 Features

- **Conversational Interface**: Natural chat to understand your preferences
- **Personalized Recommendations**: Tailored suggestions based on your tastes
- **Smart Preference Learning**: Remembers what you like/dislike during conversation
- **Comprehensive Database**: Access to detailed anime information and ratings
- **Intelligent Reasoning**: Uses ReAct pattern for thoughtful recommendations
- **Cost-Effective**: Powered by GPT-4o mini for optimal price/performance

## 🎯 How It Works

1. **Chat First**: The agent starts by getting to know your anime preferences
2. **Listen & Learn**: It understands what you've watched and what you're looking for
3. **Smart Search**: Uses tools to find anime matching your criteria
4. **Thoughtful Recommendations**: Suggests 3-5 anime with explanations of why you'll like them

## 📁 File Structure

```
Agent_csv/
├── tools.py           # LangChain tools for anime data operations
├── react_agent.py     # Conversational recommendation agent
├── demo.py           # Interactive chat interface
├── tools_test.py     # Test suite for all tools
├── requirements.txt  # Python dependencies
└── README.md        # This file
```

## 🛠️ Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY='your-openai-api-key-here'
```

### 3. Get API Key

Visit [platform.openai.com/api-keys](https://platform.openai.com/api-keys) to get your OpenAI API key.

**Note**: GPT-4o mini is perfect for conversations and very cost-effective!

## 🎯 Usage

### Interactive Conversation

Run the chat interface for personalized recommendations:

```bash
cd Agent_csv
python demo.py
```

### Programmatic Usage

```python
from react_agent import create_agent

# Create recommendation agent
agent = create_agent()

# Have a conversation
response = agent.chat("Hi! I loved Attack on Titan. What should I watch next?")
print(response)

# Continue the conversation
response = agent.chat("I prefer something recent and action-packed")
print(response)
```

## 📊 Agent's Capabilities

The agent has access to three powerful tools for recommendations:

### 1. `get_anime_ids_after_year`
- **Purpose**: Find anime from specific time periods
- **Usage**: When you mention wanting recent anime or specific years

### 2. `search_anime_ids_by_name`
- **Purpose**: Find anime by name (partial matching)
- **Usage**: When you mention specific anime or want similar shows

### 3. `get_anime_details_by_ids`
- **Purpose**: Get complete information about anime
- **Usage**: To provide detailed recommendations with ratings, genres, etc.

## 💬 Example Conversations

### Getting Started
```
You: Hi! I need some anime recommendations
Agent: Hey there! I'd love to help you find some amazing anime! 
       Tell me, what kind of shows do you usually enjoy? 
       Have you watched any anime before that you really liked?
```

### Based on Preferences
```
You: I loved Attack on Titan and Death Note
Agent: Great choices! Both are intense psychological thrillers with 
       complex stories. Are you looking for something similar with 
       dark themes and strategic elements, or maybe something different?
```

### Triggering Recommendations
```
You: I want something recent and action-packed. What do you recommend?
Agent: Perfect! Let me find some great recent action anime for you...
       [uses tools to search and recommend specific anime]
```

### For Beginners
```
You: I'm new to anime, what should I start with?
Agent: Welcome to the wonderful world of anime! Let me ask a few 
       questions to find the perfect starting point for you...
```

## 🎯 How to Get Recommendations

The agent will naturally offer recommendations, but you can also trigger them by saying:

### **🔥 FORCE COMMAND:**
- **"RECOMMEND NOW"** - Immediately get recommendations based on ALL info shared so far (bypasses any additional questions)

### **Direct Requests:**
- "What do you recommend?"
- "Show me some options"
- "Give me recommendations" 
- "What should I watch?"
- "I'm ready for suggestions"

### **Conversational Triggers:**
- "Based on what I told you, what do you think?"
- "Any ideas for me?"
- "What matches my preferences?"
- "I think you have enough info now"

### **Natural Flow:**
- The agent will proactively offer recommendations after learning about 2-3 of your preferences
- Just keep chatting about what you like, and it will suggest when it's ready!

## 🔄 Conversation Flow

```
1. Start Chat → 2. Share Preferences → 3. Type "RECOMMEND NOW" → 4. Get Instant Suggestions
     ↑                                      ↓                              ↓
     └─── Or use other triggers ──→ 3. Natural Trigger → 4. Get Suggestions → 5. Refine/Ask for More
```

## 🎭 Conversation Style

The agent is designed to be:
- **Friendly & Enthusiastic**: Uses casual language and emojis
- **Curious**: Asks engaging questions about your preferences
- **Thoughtful**: Explains WHY it recommends specific anime
- **Adaptive**: Learns from your responses during the conversation

## 🧪 Testing

### Test All Tools
```bash
cd Agent_csv
python tools_test.py
```

### Test Recommendation Agent
```bash
cd Agent_csv
python react_agent.py
```

## 🔧 Configuration

### Change OpenAI Model

```python
# Default GPT-4o mini (recommended for conversations)
agent = create_agent()

# Use GPT-4 for more complex reasoning (higher cost)
agent = create_agent("gpt-4")

# Use GPT-3.5 Turbo for basic interactions (lower cost)
agent = create_agent("gpt-3.5-turbo")
```

### Customize Conversation Style

Edit the system prompt in `react_agent.py` to modify the agent's personality:

```python
def _create_prompt(self):
    # Modify the system message to change conversation style
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Your custom personality and instructions..."),
        MessagesPlaceholder(variable_name="messages"),
    ])
    return prompt
```

## 🔍 How Recommendations Work

The agent follows this intelligent process:

1. **Conversation**: Engages naturally to understand preferences
2. **Analysis**: Processes what you tell it about your tastes
3. **Tool Selection**: Chooses appropriate search strategies
4. **Data Retrieval**: Uses tools to find matching anime
5. **Curation**: Selects the best recommendations
6. **Explanation**: Tells you WHY each recommendation fits your preferences

### Recommendation Flow

```
User Preferences → Conversation Understanding → Tool Usage → Curated Recommendations
       ↑                                                              ↓
       └─────────────── Follow-up Questions ←─────────────────────────┘
```

## 📈 Performance

- **Natural Conversations**: GPT-4o mini excels at understanding preferences
- **Fast Responses**: Optimized for interactive chat experiences
- **Smart Tool Usage**: Only searches database when needed
- **Cost-Effective**: Minimal cost per conversation (~$0.01-0.05 typically)

## 🚨 Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not found"**
   - Set your environment variable: `export OPENAI_API_KEY='your-key'`
   - Verify the key is valid and has sufficient credits

2. **Agent seems confused**
   - Try being more specific about your preferences
   - Mention anime you've watched and liked/disliked
   - Start a new conversation if context gets muddled

3. **No recommendations returned**
   - Check that the CSV file path in `tools.py` is correct
   - Verify the anime database file exists and is readable

4. **Tool errors**
   - Run `python tools_test.py` to verify all tools work correctly
   - Check pandas can read your CSV file

### Getting Better Recommendations

1. **Be Specific**: Mention exact anime titles you've enjoyed
2. **Share Context**: Tell the agent your mood or what you're looking for
3. **Give Feedback**: Let the agent know if recommendations are on track
4. **Ask Follow-ups**: Request more details or alternative suggestions

## 💰 Cost Information

**GPT-4o mini** conversation costs:
- Input: $0.15 per 1M tokens
- Output: $0.60 per 1M tokens

Typical recommendation conversation: **$0.01 - $0.05** total cost.

## 🤝 Contributing

To improve the recommendation system:

1. **Add new tools** in `tools.py` for more data access
2. **Enhance prompts** in `react_agent.py` for better conversations
3. **Improve filtering** capabilities in the existing tools
4. **Add preference memory** for returning users

## 📝 License

This project is part of the Recommender Systems suite. See main repository for license details.

---

**Ready to discover your next favorite anime? Let's chat! 🎌✨** 