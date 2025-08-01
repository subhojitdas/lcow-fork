from dataclasses import asdict, dataclass
import io
import os 
import json
from .prompt_templates import PromptTemplate, get_prompt_template
from langchain.schema import BaseMessage, SystemMessage, HumanMessage, AIMessage
from functools import partial
from typing import Optional, List, Any
import logging
from typing import Tuple
import time
import google.generativeai as genai

from langchain_community.llms import HuggingFaceHub, HuggingFacePipeline
#from langchain_community.chat_models.openai import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import BaseMessage
from langchain.chat_models.base import SimpleChatModel
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field
from transformers import pipeline
from dataclasses import dataclass
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
from transformers import GPT2TokenizerFast

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

@dataclass
class ChatModelArgs:
    """Serializable object for instantiating a generic chat model.

    Attributes
    ----------
    model_name : str
        The name or path of the model to use.
    model_url : str, optional
        The url of the model to use, e.g. via TGI. If None, then model_name or model_path must
        be specified.
    eai_token: str, optional
        The EAI token to use for authentication on Toolkit. Defaults to snow.optimass_account.cl4code's token.
    temperature : float
        The temperature to use for the model.
    max_new_tokens : int
        The maximum number of tokens to generate.
    hf_hosted : bool
        Whether the model is hosted on HuggingFace Hub. Defaults to False.
    info : dict, optional
        Any other information about how the model was finetuned.
    DGX related args
    n_gpus : int
        The number of GPUs to use. Defaults to 1.
    tgi_image : str
        The TGI image to use. Defaults to "e3cbr6awpnoq/research/text-generation-inference:1.1.0".
    ace : str
        The ACE to use. Defaults to "servicenow-scus-ace".
    workspace : str
        The workspace to use. Defaults to UI_COPILOT_SCUS_WORKSPACE.
    max_total_tokens : int
        The maximum number of total tokens (input + output). Defaults to 4096.
    """

    # model_name: str = "openai/gpt-3.5-turbo"
    model_name: str = "googleai/gemini-1.5-pro-latest"
    model_url: str = None
    temperature: float = 0.2
    top_p: float = 0.1
    max_new_tokens: int = None
    max_total_tokens: int = None
    max_input_tokens: int = None
    hf_hosted: bool = False
    info: dict = None
    n_retry_server: int = 4
    n: int=1

    def __post_init__(self):
        if self.model_url is not None and self.hf_hosted:
            raise ValueError("model_url cannot be specified when hf_hosted is True")

    def make_chat_model(self):

        '''
        if self.model_name.startswith("openai"):
            _, model_name = self.model_name.split("/")
            return ChatOpenAI(
                model_name=model_name,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )
        '''
        if self.model_name.startswith("openai"):
            _, model_name = self.model_name.split("/")
            return ChatOpenAI(
                model_name=model_name,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )
        
        elif self.model_name.startswith("anthropic"):
            _, model_name = self.model_name.split("/")
            return ChatAnthropic(
                model_name=model_name,
                anthropic_api_key=os.environ['ANTHROPIC_API_KEY'],
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )
        
        elif self.model_name.startswith("googleai"):
            os.environ["GOOGLE_API_KEY"] = "AIzaSyB2D1GSJL-f6YFPCAsl7GsUyDJaUwDqAKw"
            _, model_name = self.model_name.split("/")
            return ChatGoogleGenerativeAI(
                model=model_name, 
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )

        elif self.model_name.startswith("meta-llama"):
            return LLaMAChatModel(self.model_name)

        else:
            return HuggingFaceChatModel(
                model_name=self.model_name,
                hf_hosted=self.hf_hosted,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                max_total_tokens=self.max_total_tokens,
                max_input_tokens=self.max_input_tokens,
                model_url=self.model_url,
                n_retry_server=self.n_retry_server,
            )

    @property
    def model_short_name(self):
        if "/" in self.model_name:
            return self.model_name.split("/")[1]
        else:
            return self.model_name

    def key(self):
        """Return a unique key for these arguments."""
        return json.dumps(asdict(self), sort_keys=True)

    def has_vision(self):
        # TODO make sure to upgrade this as we add more models
        return True#"vision" in self.model_name

class LLaMAChatModel:
    def __init__(self, model_name):
        self.model_name = model_name
        import openai
        from openai import OpenAI
        self.client = OpenAI(
            api_key = os.environ['LLAMA_API_KEY'],
            base_url = "https://api.deepinfra.com/v1/openai"
            )
    
    def invoke(self, chat_messages):
        chat_messages = [
            {'role': 'system', 'content': chat_messages[0].content},
            {'role': 'user', 'content': chat_messages[1].content}
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=chat_messages,
            temperature=0.0,
            max_tokens=2000
        )
        output = response.choices[0].message
        return output
        
    


class HuggingFaceChatModel(SimpleChatModel):
    """
    Custom LLM Chatbot that can interface with HuggingFace models.

    This class allows for the creation of a custom chatbot using models hosted
    on HuggingFace Hub or a local checkpoint. It provides flexibility in defining
    the temperature for response sampling and the maximum number of new tokens
    in the response.

    Attributes:
        llm (Any): The HuggingFaceHub model instance.
        prompt_template (Any): Template for the prompt to be used for the model's input sequence.
    """

    llm: Any = Field(description="The HuggingFaceHub model instance")
    tokenizer: Any = Field(
        default=None,
        description="The tokenizer to use for the model",
    )
    prompt_template: Optional[PromptTemplate] = Field(
        default=None,
        description="Template for the prompt to be used for the model's input sequence",
    )
    n_retry_server: int = Field(
        default=4,
        description="The number of times to retry the server if it fails to respond",
    )

    def __init__(
        self,
        model_name: str,
        hf_hosted: bool,
        temperature: float,
        max_new_tokens: int,
        max_total_tokens: int,
        max_input_tokens: int,
        model_url: str,
        eai_token: str,
        n_retry_server: int,
    ):
        """
        Initializes the CustomLLMChatbot with the specified configurations.

        Args:
            model_name (str): The path to the model checkpoint.
            prompt_template (PromptTemplate, optional): A string template for structuring the prompt.
            hf_hosted (bool, optional): Whether the model is hosted on HuggingFace Hub. Defaults to False.
            temperature (float, optional): Sampling temperature. Defaults to 0.1.
            max_new_tokens (int, optional): Maximum length for the response. Defaults to 64.
            model_url (str, optional): The url of the model to use. If None, then model_name or model_name will be used. Defaults to None.
        """
        super().__init__()

        self.n_retry_server = n_retry_server

        if max_new_tokens is None:
            max_new_tokens = max_total_tokens - max_input_tokens
            logging.warning(
                f"max_new_tokens is not specified. Setting it to {max_new_tokens} (max_total_tokens - max_input_tokens)."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if isinstance(self.tokenizer, GPT2TokenizerFast):
            # TODO: make this less hacky once tokenizer.apply_chat_template is more mature
            logging.warning(
                f"No chat template is defined for {model_name}. Resolving to the hard-coded templates."
            )
            self.tokenizer = None
            self.prompt_template = get_prompt_template(model_name)

        if temperature < 1e-3:
            logging.warning(
                "some weird things might happen when temperature is too low for some models."
            )

        model_kwargs = {
            "temperature": temperature,
        }

        if model_url is not None:
            logging.info("Loading the LLM from a URL")
            client = InferenceClient(model=model_url, token=eai_token)
            self.llm = partial(
                client.text_generation, temperature=temperature, max_new_tokens=max_new_tokens
            )
        elif hf_hosted:
            logging.info("Serving the LLM on HuggingFace Hub")
            model_kwargs["max_length"] = max_new_tokens
            self.llm = HuggingFaceHub(repo_id=model_name, model_kwargs=model_kwargs)
        else:
            logging.info("Loading the LLM locally")
            pipe = pipeline(
                task="text-generation",
                model=model_name,
                device_map="auto",
                max_new_tokens=max_new_tokens,
                model_kwargs=model_kwargs,
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None or run_manager is not None or kwargs:
            logging.warning(
                "The `stop`, `run_manager`, and `kwargs` arguments are ignored in this implementation."
            )

        if self.tokenizer:
            messages_formated = _convert_messages_to_dict(messages)
            prompt = self.tokenizer.apply_chat_template(messages_formated, tokenize=False)

        elif self.prompt_template:
            prompt = self.prompt_template.construct_prompt(messages)

        itr = 0
        while True:
            try:
                response = self.llm(prompt)
                return response
            except Exception as e:
                if itr == self.n_retry_server - 1:
                    raise e
                logging.warning(
                    f"Failed to get a response from the server: \n{e}\n"
                    f"Retrying... ({itr+1}/{self.n_retry_server})"
                )
                time.sleep(5)
                itr += 1

    def _llm_type(self):
        return "huggingface"


def _convert_messages_to_dict(messages):
    """
    Converts a list of message objects into a list of dictionaries, categorizing each message by its role.

    Each message is expected to be an instance of one of the following types: SystemMessage, HumanMessage, AIMessage.
    The function maps each message to its corresponding role ('system', 'user', 'assistant') and formats it into a dictionary.

    Args:
        messages (list): A list of message objects.

    Returns:
        list: A list of dictionaries where each dictionary represents a message and contains 'role' and 'content' keys.

    Raises:
        ValueError: If an unsupported message type is encountered.

    Example:
        >>> messages = [SystemMessage("System initializing..."), HumanMessage("Hello!"), AIMessage("How can I assist?")]
        >>> _convert_messages_to_dict(messages)
        [
            {"role": "system", "content": "System initializing..."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "How can I assist?"}
        ]
    """

    # Mapping of message types to roles
    message_type_to_role = {
        SystemMessage: "system",
        HumanMessage: "user",
        AIMessage: "assistant",
    }

    chat = []
    for message in messages:
        message_role = message_type_to_role.get(type(message))
        if message_role:
            chat.append({"role": message_role, "content": message.content})
        else:
            raise ValueError(f"Message type {type(message)} not supported")

    return chat
