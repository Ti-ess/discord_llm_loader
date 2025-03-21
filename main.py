import torch
from transformers import pipeline, AutoTokenizer, AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import discord
import json
import pprint
import os
import configparser
import sys
import requests

def loadConfig() -> configparser.ConfigParser: #load the config file
    config = configparser.ConfigParser()
    default = {
        "DISCORD": {
            "BOT_TOKEN" : "None",
            "GUILD_ID" : "None",
            "SUPER_USER_ID" : "None",
            "BOT_ID" : "None"
        },
        "CHANNELS": {},
        "MODEL": {
            "MODEL_NAME" : "None",
            "ALLOW_CPU" : "False",
            "DEFAULT_SNIPPET" : "None"
        }
    }
    config.read('config.ini')
    if not config.sections():
        config.read_dict(default)
        with open('config.ini', 'w') as f:
            config.write(f)
    if not validateConfig(config):
        sys.exit()
    return config
    
#validate that a valid config file was loaded
def validateConfig(config: configparser.ConfigParser) -> bool:
    if config['DISCORD']['BOT_TOKEN'] == "None":
        print("No Discord token found in config.ini. Please add your bot's token to the config file.")
        return False
    if config['DISCORD']['GUILD_ID'] == "None":
        print("No guild ID found in config.ini. Please add the ID of the guild you want the bot to connect to.")
        return False
    if config['DISCORD']['SUPER_USER_ID'] == "None":
        print("No super user ID found in config.ini. Please add the ID of the user you want to have special permissions.")
        return False
    if config['DISCORD']['BOT_ID'] == "None":
        print("No bot ID found in config.ini. Please add the ID of the bot user.")
        return False
    if not config['CHANNELS'] or config['CHANNELS'] == []:
        print("No channel IDs found in config.ini. Please add the IDs of the channels you want the bot to listen to.")
        return False
    if config['MODEL']['MODEL_NAME'] == "None":
        print("No model name found in config.ini. Please add the name of the model you want the bot to use.")
        return False
    if config['MODEL']['DEFAULT_SNIPPET'] == "None":
        print("No default snippet found in config.ini. Please add a default snippet for the bot to use.")
        return False
    return True

#set variables from config
config = loadConfig()
TOKEN = config['DISCORD']['token']
GUILD = int(config['DISCORD']['GUILD_ID'])
SUPER_USER_ID = int(config['DISCORD']['SUPER_USER_ID'])
BOT_ID = int(config['DISCORD']['BOT_ID'])
channelIDs = config['CHANNELS'].keys()
for i in range(len(channelIDs)):
    channelIDs[i] = int(channelIDs[i])
model_alias = config['MODEL']['MODEL_NAME']
allowCPU = config['MODEL'].getboolean('ALLOW_CPU')
defalut_snippet = config['MODEL']['DEFAULT_SNIPPET']
chatHistory = []

def failGPU(): #This function is called when the bot fails to connect to the GPU. It will print an error message and exit the script if the user has not enabled CPU mode in the config file.
    print("FAILED TO CONNECT TO GPU")
    print("Please ensure that your GPU is enabled and that you have the necessary drivers installed.")
    if not allowCPU:
        print("If you don't have a GPU, you can enable CPU mode in the config.ini file.")
        sys.exit()

os.environ['HF_HOME'] = './models'
device = "cuda" if torch.cuda.is_available() else failGPU()

if device == "cpu":
    print("We're using our CPU for inference. This will be slow.")

#list of models. Extend by adding more models to the list
models = {
    "llama_31" : "meta-llama/Llama-3.1-8B-Instruct",
    "llama_32" : "meta-llama/Llama-3.2-3B-Instruct",
    "qwen" : "Qwen/Qwen2.5-1.5B-Instruct",
    "fimbulvetr" : "Sao10K/Fimbulvetr-11B-v2",
    "mistral" : "mistralai/Mistral-7B-Instruct-v0.3",
    "capybara" : "TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ",
    "llamaUnc" : "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2",
    "deepSeek" : "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "gemma" : "google/gemma-3-4b-it"
}

#default prompt. Created using the default snippet from the config file.
prompt = [
    {"role": "system", "content": 
    [
        {"type":"text", "text": defalut_snippet},
    ]}]

defalut_prompt = prompt

if model_alias in models:
    active_model = models[model_alias]
else:
    print("Model alias not in list. Attempting to load based on HF repo name.")
    active_model = model_alias

#Different models need different tokenizers. This block of code sets the tokenizer based on the model alias.
if model_alias == "fimbulvetr":
    tokenizer = AutoTokenizer.from_pretrained(active_model)
    tokenizer.chat_template = """
    System {{ Prompt}}
    USER: {{ Input }}
    ASSISTANT:
    """
    generator = pipeline("text-generation", model=active_model, device=device, torch_dtype=torch.float16, tokenizer=tokenizer)
elif model_alias == "gemma":
    generator = pipeline(
    "image-text-to-text",
    model="google/gemma-3-4b-it",
    device="cuda",
    use_fast=True,
    torch_dtype=torch.bfloat16
)
else:
    generator = pipeline(model=active_model, device=device, torch_dtype=torch.bfloat16)

def trimResponse(response): #Removes the think tag from the response if using a deep thinking model. Also splits the response into multiple messages if it is too long.
    r = ""
    if model_alias == "deepSeek":
        r = response.split("</think>")[1].strip()
    else:
        r = response
    print("THE BOT SAYS: " + r)
    if len(r) > 2000:
        i = 0
        n = []
        while i < len(r) and len(r[i:] > 2000):
            n.append(r[i:i+2000])
            i += 2000
        n.append(r[i:])
        return n
    return [r]

async def createRequest(user_message, character="user", attachment=None): #Meat and potatoes of the bot. This is where the model is called and the response is generated.
    global prompt
    if model_alias == "gemma":
        model_id = "google/gemma-3-4b-it"
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, device_map="auto"
        ).eval()

        # Format prompt for Gemma 3
        if attachment is not None:
            prompt.append({"role": character, "content": [
                {"type": "text", "text": user_message},
                {"type": "image", "url": attachment}
            ]})
        else:
            prompt.append({"role": character, "content": [
                {"type": "text", "text": user_message}
            ]})
        print("THE PROMPT IS: " + str(prompt))
        processor = AutoProcessor.from_pretrained(model_id)
        inputs = processor.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        generationPrime = None
        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=2000, do_sample=False)
            generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
        if len(prompt) > 10:
            prompt = prompt[-10:]
    else:
        # Handle other models
        prompt.append({"role": character, "content": user_message})
        if len(prompt) > 10:
            prompt = prompt[-10:]
            
        if model_alias == "fimbulvetr":
            generation = generator(
                prompt,
                do_sample=True,
                temperature=0.6,
                top_p=6,
                max_new_tokens=2000
            )
        else:
            generation = generator(
                prompt,
                do_sample=True,
                temperature=0.6,
                top_p=6,
                max_new_tokens=2000
            )
    if model_alias == "gemma":
        newLine = decoded
    else:
        newLine = generation[0]['generated_text']
    if model_alias == "gemma":
        prompt.append({"role": "assistant", "content": [{"type":"text","text":newLine}]})
    else:
        prompt.append({"role": "assistant", "content": newLine})
    saveChatHistory(prompt)
    pprint.pprint(prompt)
    return newLine

def saveChatHistory(prompt):
    with open('chatHistory.json', 'w') as f:
        json.dump(prompt, f)

client = discord.Client(intents=discord.Intents(messages=True, guilds=True, message_content=True, members=True))

@client.event
async def on_ready(): #This is the event that triggers when the bot connects to Discord.
    global prompt
    print(f'{client.user.name} has connected to USER!')
    botSay = await createRequest("The bot awakens from its slumber, ready for another round of conversations.", "user")
    for channel in channelIDs:
        channel = client.get_channel(channel)
        print("BOTSAY WAS: " + botSay)
        await channel.send(trimResponse(botSay)[0])
        prompt = prompt[:-1]


@client.event
async def on_message(message): #This is the event that triggers when the bot receives a message.
    global prompt
    #print(str(type(message.author)))
    #print("AUTHOR:" + str(dir(message.author)))
    authorUsername = message.author.nick
    if message.channel.id in channelIDs and message.author.id == SUPER_USER_ID and message.content.startswith("power word"): #This is the super user command. It allows the super user to reset the chat history or put the bot to sleep by saying things in chat.
        if message.content == "power word reset":
            prompt = defalut_prompt
            await message.reply("**[Chat history reset]**")
        elif message.content == "power word sleep":
            botsSay = await createRequest("The bot says goodnight to everyone.", "system")
            utterance = trimResponse(botsSay)
            if len(utterance.replace(" ", "")) <= 1:
                utterance = "zzz"
            await message.reply(utterance)
            #shutdown script
            quit()
    elif message.channel.id in channelIDs and message.author.id != BOT_ID: #This is the main chat event. It triggers when the bot receives a message from a user.
        async with message.channel.typing():
            attachment = None
            if message.attachments:
                attachment = message.attachments[0].url
            print("\nMESSAGE: " + message.content + "\nATTACHMENT: " + str(attachment))
            content = message.content
            #print(botSay)
            utterance = await createRequest(content, "user", attachment)
            utterance = trimResponse(utterance)
        for u in utterance:
            await message.reply(u)
        print("MESSAGE SENT")

print("Starting the bot...")
client.run(TOKEN)
