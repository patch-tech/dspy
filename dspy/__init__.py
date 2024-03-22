"""Demonstrate-Search-Predicate: a Python framework for programming, not prompting, large language models."""

from dsp import utils as dsp_utils
from dsp.modules.anthropic import Claude
from dsp.modules.azure_openai import AzureOpenAI
from dsp.modules.bedrock import Bedrock
from dsp.modules.clarifai import ClarifaiLLM
from dsp.modules.cohere import Cohere
from dsp.modules.colbertv2 import ColBERTv2
from dsp.modules.databricks import Databricks
from dsp.modules.google import Google
from dsp.modules.gpt3 import GPT3
from dsp.modules.hf import HFModel
from dsp.modules.hf_client import (
    Anyscale,
    ChatModuleClient,
    HFClientSGLang,
    HFClientTGI,
    HFClientVLLM,
    HFServerTGI,
    Together,
)
from dsp.modules.llama_cpp import LlamaCpp
from dsp.modules.ollama import OllamaLocal
from dsp.modules.pyserini import PyseriniRetriever
from dsp.modules.sbert import SentenceTransformersCrossEncoder
from dsp.modules.sentence_vectorizer import (
    BaseSentenceVectorizer,
    CohereVectorizer,
    NaiveGetFieldVectorizer,
    OpenAIVectorizer,
    SentenceTransformersVectorizer,
)
from dspy.predict import retry
from dspy.predict.aggregation import majority
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.predict.chain_of_thought_with_hint import ChainOfThoughtWithHint
from dspy.predict.knn import KNN
from dspy.predict.multi_chain_comparison import MultiChainComparison
from dspy.predict.predict import Predict
from dspy.predict.program_of_thought import ProgramOfThought
from dspy.predict.react import ReAct
from dspy.predict.retry import Retry
from dspy.primitives import *
from dspy.retrieve import *
from dspy.signatures import *

# Functional must be imported after primitives, predict and signatures
from dspy.functional import *  # isort: skip


settings = dsp_utils.settings
configure = settings.configure
context = settings.context

OpenAI = GPT3
