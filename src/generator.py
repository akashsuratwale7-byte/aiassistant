
from langchain_core.prompts import PromptTemplate
from src.retriever import retrieve_docs
from src.config import MODEL_PROVIDER, OPENAI_API_KEY

# Hugging Face imports
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline

# OpenAI imports
from langchain_openai import ChatOpenAI


prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert travel assistant.
Use the provided travel guide context to answer the question.

CRITICAL INSTRUCTIONS:
- You MUST extract and list every single day requested.
- Use markdown bullet points for each day.
- When you finish the itinerary, you MUST write the exact word "[END]". Do not write anything after this word.

Context:
{context}

Question:
{question}

Answer:"""
)

if MODEL_PROVIDER == "huggingface":
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.1,
        top_k=50,
        top_p=0.95,
        return_full_text=False,
        repetition_penalty=1.05,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
elif MODEL_PROVIDER == "openai":
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=0.7)
else:
    raise ValueError(f"Unknown MODEL_PROVIDER: {MODEL_PROVIDER}")


qa_chain = prompt_template | llm

def generate_answer(query: str) -> str:
    docs = retrieve_docs(query)
    context = "\n".join(docs)
    result = qa_chain.invoke({"context": context, "question": query})
    # If using OpenAI, return only the 'content' field
    if MODEL_PROVIDER == "openai":
        #print(result.content)
        return result.content
    return result
