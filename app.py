import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch
from pydantic import BaseModel, Field

# -------------------------------
# Page
# -------------------------------
st.set_page_config(page_title="Prompt Quality Agent (LangChain)", layout="wide")
st.title("🧠 Prompt Quality Agent — LangChain Pipeline")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("⚙️ Settings")

api_key = st.sidebar.text_input("Gemini API Key", type="password")

model = st.sidebar.selectbox(
    "Select Gemini Model",
    [
        "models/gemini-2.5-flash",
        "models/gemini-1.5-pro"
    ]
)

temperature = st.sidebar.slider("Creativity", 0.0, 1.0, 0.2)
threshold = st.sidebar.slider("Minimum Acceptable Score", 1, 10, 7)

# -------------------------------
# LLM
# -------------------------------
def get_llm():
    return ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model=model,
        temperature=temperature
    )

# -------------------------------
# SCHEMA (Structured Output)
# -------------------------------
class Evaluation(BaseModel):
    score: int = Field(description="Score from 0 to 10")
    reason: str
    missing: str
    verdict: str

parser = PydanticOutputParser(pydantic_object=Evaluation)

# -------------------------------
# PROMPT TEMPLATES
# -------------------------------
eval_prompt = PromptTemplate.from_template(
"""
You are a PROMPT QUALITY EVALUATOR.

Evaluate based on:
- clarity
- specificity
- context
- structure

Return ONLY JSON.

{format_instructions}

Prompt:
{prompt}
"""
)

improve_prompt = PromptTemplate.from_template(
"""
You are a PROMPT IMPROVEMENT AGENT.

Fix the following issues:
{missing}

Keep intent same.

Prompt:
{prompt}
"""
)

# -------------------------------
# CHAINS
# -------------------------------
def build_chains(llm):

    # Evaluator chain
    evaluator_chain = (
        eval_prompt.partial(format_instructions=parser.get_format_instructions())
        | llm
        | parser
    )

    # Improver chain
    improver_chain = improve_prompt | llm

    # Decision function
    def decision_fn(data):
        if data["evaluation"].score < threshold:
            return "improve"
        return "accept"

    # Combine evaluation with original input
    def attach_input(evaluation, original):
        return {
            "evaluation": evaluation,
            "original": original
        }

    # Branch logic
    branch = RunnableBranch(
        (lambda x: x["evaluation"].score < threshold,
         RunnableLambda(lambda x: {
             "final": improver_chain.invoke({
                 "prompt": x["original"],
                 "missing": x["evaluation"].missing
             }).content,
             "evaluation": x["evaluation"]
         })),
        RunnableLambda(lambda x: {
            "final": x["original"],
            "evaluation": x["evaluation"]
        })
    )

    return evaluator_chain, branch

# -------------------------------
# UI
# -------------------------------
user_prompt = st.text_area("Enter a prompt")

run = st.button("🚀 Analyze Prompt")

# -------------------------------
# MAIN
# -------------------------------
if run:

    if not api_key:
        st.error("Enter API key")
        st.stop()

    if not user_prompt:
        st.error("Enter a prompt")
        st.stop()

    llm = get_llm()

    evaluator_chain, branch = build_chains(llm)

    with st.spinner("Running LangChain pipeline..."):

        # Step 1: Evaluate
        evaluation = evaluator_chain.invoke({"prompt": user_prompt})

        # Step 2: Attach original input
        state = {
            "evaluation": evaluation,
            "original": user_prompt
        }

        # Step 3: Branch (decision + improvement)
        result = branch.invoke(state)

    # -------------------------------
    # OUTPUT
    # -------------------------------
    st.subheader("📊 Evaluation")

    st.metric("Score", f"{evaluation.score}/10")
    st.progress(evaluation.score / 10)

    st.subheader("🧠 Reason")
    st.write(evaluation.reason)

    st.subheader("⚠️ Missing")
    st.write(evaluation.missing)

    if evaluation.score < threshold:
        st.warning("⚠️ Prompt needs improvement")
    else:
        st.success("✅ Prompt is good")

    st.subheader("✨ Final Prompt")
    st.success(result["final"])
