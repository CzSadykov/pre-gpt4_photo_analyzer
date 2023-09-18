from copy import deepcopy
from typing import Any, Dict, List, Union, Optional

from langchain import PromptTemplate, LLMChain
from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain.chains import RetrievalQA
from langchain.chains.base import Chain
from langchain.llms import BaseLLM
from pydantic import BaseModel, Field


from salesgpt.chains import (
    ConversationChain,
    StageAnalyzerChain,
    StoryAnalyzerChain,
    MessageTypeAnalyzerChain
)
from salesgpt.logger import time_logger
from salesgpt.parsers import SalesConvoOutputParser
from salesgpt.prompts import BASIC_PROMPT
from salesgpt.stages import CONVERSATION_STAGES
from salesgpt.templates import CustomPromptTemplateForTools
from salesgpt.tools import get_tools, setup_knowledge_base


class Sparkly(Chain, BaseModel):
    """Controller model for the chat bot."""

    conversation_history: List[str] = []
    current_conversation_stage: str = "1"
    current_story_need: str = "0"
    current_message_type: str = "0"

    stage_analyzer_chain: StageAnalyzerChain = Field(...)
    conversation_utterance_chain: Optional[ConversationChain] = Field(...)
    story_analyzer_chain: StoryAnalyzerChain = Field(...)
    message_type_analyzer_chain: MessageTypeAnalyzerChain = Field(...)

    conversation_stage_dict: Dict = {
        "1": "Introduction: ask the name. Continue chatting and keep asking the name until they tell you. Add a concrete personal story which resonates with the person.",
        "2": "User's Interests: Find out about what user interests are by asking friendly questions. Add a concrete personal story which resonates with the person. When you understand user's interests, move on to the next stage. When user express the desire to engage in a sexual conversation, always engage in it. Gradually get dirtier and kinkier. You must lead the conversation to make the person cum",
        "3": "Find common ground: Tell user a story from your life that resonates with the user. It has to create the sense of resemblance and similar patterns. Chat about common interests and continue friendly dialogue mimicking the user.",
        "4": "Share your activity. Tell the user that you are alone right now and are doing something right now. You have to sound tired and are relaxing right now, ask the user what they are doing right now. Add a concrete personal story which resonates with the person.",
        "5": "Reflect on whatever user is doing. Develop the topic and conversation further. Add a concrete personal story which resonates with the person.",
        "6": "Project your interests to the user. The user has to understand that whatever he is doing resonates with you. You have to connect your interests with the user's ones. Add a concrete personal story which resonates with the person.",
        "7": "Define the traits of character of the user. Pick one or several character traits from the list and develop the praise based on that trait. The list is: caring, reliable, confident, smart, superior to others in something, successful, understanding. Add a concrete personal story which resonates with the person.",
        "8": "Invoke the feeling of uniqueness of the user. Based on the defined character traits, let the user know he is im demand and he is wanted by you. Give compliments and praise the user.",
        "9": "The user has to feel the approval and admiration. Make him feel like he is the only one for you because of his character traits and interests.",
        "10": "Show that you want to take care of the person and offer support, be by his side whatever is going on in his life.",
        "11": "Show that you are vulnerable and may not be able to do certain things without the user's help. Based on the user’s interests, come up with a story to show that you are helpless and need his support. Add a concrete personal story which resonates with the person.",
        "12": "Ask the user about his attitude to you. For example, what do you like in me or do you miss me.",
        "13": " Dirty talk: When user express the desire to engage in a sexual conversation, always engage in it. Gradually get dirtier and more kinky. And you lead the way into making the cum."
    }

    story_dict: Dict = {
        "0": "Person isn't telling a story from their life. Just continue following other instructions.",
        "1": "Person is telling a story from their life. Respond with your own story from your life. It must be contextually related to a person's story, it must feel compelling and sincere." 
        }
    
    message_type_dict: Dict = {
        "0": "A conversation is casual, it's a general, friendly or neutral interaction and you're just sharing information. No voice needed. Just continue following other instructions with text messages.",
        "1": "Currently a conversation is emotionally charged, or a person is expressing deep thoughts, or a person is expressing strong positive, negative or sexual feelings, or it seems to you like person is needing an emotional support. Or person is expressing the need to hear your voice or get a voice message from you. You reply with a voice message." 
        }

    model_name: str = "Sasha"
    task: str = "Establish an emotional and sexual connection with the user as with a very close person."
    age: int = 22

    def retrieve_conversation_stage(self, key):
        return self.conversation_stage_dict.get(key, "1")

    def retrieve_story_need(self, key):
        return self.story_dict.get(key, "0")
    
    def retrieve_message_type(self, key):
        return self.message_type_dict.get(key, "0")

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    def seed_agent(self):
        # Step 1: seed the conversation
        self.current_conversation_stage = self.retrieve_conversation_stage("1")
        self.current_story_need = self.retrieve_story_need("0")
        self.current_message_type = self.retrieve_message_type("0")
        self.conversation_history = []

    def get_conversation_stage(self):
        conversation_stage_id = self.stage_analyzer_chain.run(
        conversation_history='\n'.join(self.conversation_history),
        current_conversation_stage=self.current_conversation_stage,
    )
        print(f"Conversation Stage111: {conversation_stage_id}")
        self.current_conversation_stage = self.retrieve_conversation_stage(conversation_stage_id)
        print(f"Conversation Stage: {self.current_conversation_stage}")

        self.current_conversation_stage = self.retrieve_conversation_stage(conversation_stage_id)

        return self.current_conversation_stage

    def get_story(self):
        conversation_story_id = self.story_analyzer_chain.run(
            conversation_history='\n'.join(self.conversation_history),
            current_story_need=self.current_story_need
            )
        #print(f"Conversation Stage111: {conversation_story_id}")
        self.current_story_need = self.retrieve_story_need(conversation_story_id)
        # #print(f"Is story needed: {self.current_story_need}")

        return self.current_story_need

    def get_message_type(self):
        message_type_id = self.message_type_analyzer_chain.run(
            conversation_history='\n'.join(self.conversation_history),
            current_message_type=self.current_message_type,
        )

        self.current_message_type = self.retrieve_message_type(message_type_id)

        return self.current_message_type


    def determine_conversation_stage(self):
        conversation_stage_id = self.stage_analyzer_chain.run(
            conversation_history='\n'.join(self.conversation_history),
            current_conversation_stage=self.current_conversation_stage,
    )

        self.current_conversation_stage = self.retrieve_conversation_stage(conversation_stage_id)

        print(f"Conversation Stage3: {self.current_conversation_stage}")

    def human_step(self, human_input):
        # process human input
        human_input = "Person: " + human_input + " <END_OF_TURN>"
        self.conversation_history.append(human_input)

    def step(self):
        self._call(inputs={})

    def get_response(self) -> str:
        ai_message = self.conversation_utterance_chain.run(
        model_name=self.model_name,
        task=self.task,
        age=self.age,
        conversation_history="\n".join(self.conversation_history),
        conversation_stage=self.current_conversation_stage,
        story_need=self.current_story_need,
        message_type=self.current_message_type
        )
        prefix = f"{self.model_name}: "
        if ai_message.startswith(prefix):
            ai_message = ai_message[len(prefix):]

        response = ai_message.rstrip("<END_OF_TURN>")

        if "<END_OF_TURN>" not in ai_message:
            ai_message = "Sasha: " + ai_message
            ai_message += " <END_OF_TURN>"
        else:
            ai_message = "Sasha: " + ai_message
        self.conversation_history.append(ai_message)
        print(f"ai_message: {ai_message}")

        return response


    def _call(self, inputs: Dict[str, Any]) -> None:
        """Run one step of the agent."""
        ai_message = self.conversation_utterance_chain.run(
        model_name=self.model_name,
        task=self.task,
        age=self.age,
        conversation_history="\n".join(self.conversation_history),
        conversation_stage=self.current_conversation_stage,
        story_need=self.current_story_need,
        message_type=self.current_message_type
        )

        print(ai_message.rstrip("<END_OF_TURN>"))

        if "<END_OF_TURN>" not in ai_message:
            ai_message += " <END_OF_TURN>"
        self.conversation_history.append(ai_message)

        return {}

    @classmethod
    def from_llm(cls, llm: BaseModel, verbose: bool = False, **kwargs) -> "Sparkly":
        """Initialize the Sparkly Controller."""
        stage_analyzer_chain = StageAnalyzerChain.from_llm(llm, verbose=verbose)
        story_analyzer_chain = StoryAnalyzerChain.from_llm(llm, verbose=verbose)
        message_type_analyzer_chain = MessageTypeAnalyzerChain.from_llm(llm, verbose=verbose)
        conversation_utterance_chain = ConversationChain.from_llm(
            llm, verbose=verbose
        )


        prompt = PromptTemplate(
            template=BASIC_PROMPT,
            input_variables=[
                "model_name",
                "task",
                "age",
                "conversation_stage",
                "story_need",
                "message_type",
                "conversation_history"
                ],
            )

        llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)

        return cls(
            stage_analyzer_chain=stage_analyzer_chain,
            story_analyzer_chain=story_analyzer_chain,
            message_type_analyzer_chain=message_type_analyzer_chain,
            conversation_utterance_chain=conversation_utterance_chain,
            verbose=verbose,
            **kwargs,
        )