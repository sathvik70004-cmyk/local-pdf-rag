from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from models import llm
from store import search
from config import TOP_K

prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the context below. "
    "If the answer is not in the context, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)

chain = prompt | llm | StrOutputParser()               # fill prompt -> Gemma -> plain text


def ask(question):
    results = search(question, TOP_K)                  # [(Document, distance), ...]
    docs = [d for d, _ in results]
    dists = [s for _, s in results]
    context = "\n\n".join(
        f"[{d.metadata['source']}, page {d.metadata.get('page', '?')}] {d.page_content}"
        for d in docs
    )
    answer = chain.invoke({"context": context, "question": question})
    return answer, docs, dists