# Discord LLM Loader

A Discord bot that integrates with Large Language Models (LLMs) to interact with users in chat and process images.

## Features

- Load and run LLMs directly within your Discord server
- Chat with users using natural language processing
- Process and analyze images shared in Discord channels
- Customizable responses and behavior

## Installation

```bash
# Clone the repository
git clone https://github.com/Ti-ess/discord_llm_loader.git
cd discord_llm_loader

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1a. Open the file config.ini and fill out the relevant information

```[DISCORD]
BOT_TOKEN=PUT_BOT_TOKEN_HERE
GUILD_ID=PUT_GUILD_ID_HERE
SUPER_USER_ID=PUT_SUPER_USER_ID_HERE
BOT_ID=PUT_BOT_ID_HERE

[CHANNELS]
PUT_CHANNEL_ID_HERE
PUT_CHANNEL_ID_HERE

[MODEL]
ALLOW_CPU=False
MODEL_NAME=gemma
DEFAULT_SNIPPET=The following is a conversation between a helpful AI assistant and their users. This conversation takes place within a discord server.
```

1b. You may need to set your huggingface token in HuggingfaceCLI to download models automatically

For more information about Hugging Face models and tokens, visit [Hugging Face](https://huggingface.co/).

```bash
# To use Hugging Face models, set your access token:
# 1. Create an account at https://huggingface.co if you don't have one
# 2. Go to https://huggingface.co/settings/tokens
# 3. Create a new token with read access
# 4. Set the token in your environment:

export HUGGINGFACE_TOKEN=your_token_here

# Or on Windows:
# set HUGGINGFACE_TOKEN=your_token_here

# You can also add this to your .bashrc or .zshrc file for persistence
```

2. Obtain a Discord bot token:
    - Go to the [Discord Developer Portal](https://discord.com/developers/applications)
    - Create a new application
    - Go to the Bot section and create a bot
    - Copy the token to your `.env` file

## Usage

```bash
# Start the bot
python main.py
```

## The creature LIVES

The bot will respond to messages from any channels listed in the \[CHANNELS\] section of the config file. Have fun.

## License

MIT