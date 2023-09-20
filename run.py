import os
import json
import telebot

from langchain.chat_models import ChatOpenAI
from salesgpt.agents import Sparkly

from salesgpt.photo_analyzer import PhotoAnalyzer
from salesgpt.photo_questions import PhotoQuestionsGenerator
from salesgpt.photo_summary import PhotoSummarizer

os.environ['OPENAI_API_KEY'] = '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'

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

questions = [
    "Are there any humans on the photo?",
    "How many humans are on the photo?",
    "What object is on the photo?",
    "Is it indoors or outdoors?",
    "Is it day or night?",
    "What is the mood of the photo?",
    "Is there a face on the photo?",
    "Is there a penis on the photo?",
    "Are you sure it's a penis?",
    "Is there a sexual content on the photo?"
    ]

iterations = 2

# llm = CerebriumAI(
# endpoint_url="https://run.cerebrium.ai/v2/p-fa7750f1/lamallvm127/predict"
# )

llm_1 = ChatOpenAI(temperature=0.4, model_name='gpt-3.5-turbo')
llm_2 = ChatOpenAI(temperature=0.1, model_name='gpt-3.5-turbo')

analyzer = PhotoAnalyzer()
questions_generator = PhotoQuestionsGenerator.from_llm(llm_2)
summary_generator = PhotoSummarizer.from_llm(llm_2)

with open("examples/test_config.json", "r") as file:
    config = json.load(file)

agent = Sparkly.from_llm(llm_1, verbose=True, **config)

agent.seed_agent()

user_data = {}

bot = telebot.TeleBot('6461197197:AAH9UDhccFZxrOdgAZBADXY0UgK4XTl_WnY')


@bot.message_handler(commands=["start"])
def start(message):
    bot.reply_to(message, "Try me!")


@bot.message_handler(content_types=['text'])
def handle_text(message):
    user_input = message.text
    agent.human_step(user_input)

    response = agent.get_response()
    stage = agent.get_conversation_stage()
    story_need = agent.get_story()
    message_type = agent.get_message_type()

    if response:
        bot.send_message(message.chat.id, response)
    else:
        bot.send_message(message.chat.id, "Oops, time for an awkward silence")

    if response:
        bot.send_message(message.chat.id, f'CONVERSATION STAGE: {stage}')
    else:
        bot.send_message(message.chat.id, "Stage not determined")

    if response:
        bot.send_message(message.chat.id, f'IS STORY NEEDED: {story_need}')
    else:
        bot.send_message(message.chat.id, "Need in story not determined")

    if response:
        bot.send_message(
            message.chat.id, f'REQUIRED MESSAGE TYPE: {message_type}'
        )
    else:
        bot.send_message(message.chat.id, "Message type not determined")

    print("--------------------")


def save_photo(message):
    if not os.path.exists('photos'):
        os.makedirs('photos')
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    photo_path = os.path.join("photos", f"{message.photo[-1].file_id}.jpg")
    with open(photo_path, 'wb') as new_file:
        new_file.write(downloaded_file)
    return photo_path


@bot.message_handler(content_types=['photo'])
def handle_photo(message):

    photo_info = analyzer.analyze_photo(message, bot, questions)
    photo_qa = analyzer.format_photo_info(photo_info)

    print("Before questions generation:", photo_qa)

    mod_questions = []

    for i in range(iterations):
        print("Iteration: ", i)
        new_questions = [i for i in questions_generator.create(input=photo_qa)]
        mod_questions.extend(new_questions)
        print("New questions for iteration: ", mod_questions)
        new_qa = analyzer.analyze_photo(message, bot, mod_questions[-5:])
        print("New QA: ")
        photo_qa += analyzer.format_photo_info(new_qa)
        print("QA after the iteration: ", photo_qa)

    photo_summary = summary_generator.create(input=photo_qa)

    user_input = """
    Take a deep breath and react to my photo with a simple lexicon,
    in a positive manner. Photo:
    """ + photo_summary
    agent.human_step(user_input)

    response = agent.get_response()

    if response:
        bot.send_message(message.chat.id, response)
    else:
        bot.send_message(message.chat.id, "Oops, time for an awkward silence")

    # if response:
    #     bot.send_message(message.chat.id, f'{photo_qa}')
    # else:
    #     bot.send_message(message.chat.id, "QA not determined")

    if response:
        bot.send_message(message.chat.id, f'Photo summary: {photo_summary}')
    else:
        bot.send_message(message.chat.id, "Photo summary not determined")


bot.polling()
