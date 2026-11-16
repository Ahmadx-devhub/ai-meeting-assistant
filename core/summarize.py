
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()



def get_llm():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file")

    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=2048,
        groq_api_key=api_key,
        max_retries=2,
        timeout=60
    )


def split_transcript(transcript: str) -> list[str]:

    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    return splitter.split_text(transcript)



def summarize(transcript: str) -> str:

    llm = get_llm()


    map_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an expert meeting summarizer.

            Summarize this portion of a meeting transcript.
            
            Instructions:
            - Extract important discussion points.
            - Preserve names, dates and important details.
            - Identify decisions and action items.
            - Do not invent information.
            - Keep the summary concise.
            """
        ),
        ("human", "{text}")
    ])

    map_chain = map_prompt | llm | StrOutputParser()


    chunks = split_transcript(transcript)

    
    chunk_summaries = [
        map_chain.invoke({"text": chunk})
        for chunk in chunks
    ]


    combined_text = "\n\n".join(chunk_summaries)


    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an expert meeting summarizer.

            Combine the following partial summaries
            into one professional meeting summary.

            Structure your response using:

            1. Meeting Overview
            2. Important Discussion Points
            3. Key Decisions
            4. Action Items
            5. Conclusion

            Instructions:
            - Remove repeated information.
            - Preserve important details.
            - Do not invent information.
            - Use clear headings and bullet points.
            - Maintain a logical flow.
            """
        ),
        ("human", "{text}")
    ])

    combined_chain = combined_prompt | llm | StrOutputParser()

    return combined_chain.invoke({
        "text": combined_text
    })



def generate_title(transcript: str) -> str:

    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty")

    llm = get_llm()

    title_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a professional meeting title generator.

            Based on the meeting transcript, generate
            one short and professional meeting title.

            Rules:
            - Maximum 8 words.
            - Clearly represent the main meeting topic.
            - Do not use quotation marks.
            - Return only the title, nothing else.
            """
        ),
        ("human", "{text}")
    ])

    title_chain = title_prompt | llm | StrOutputParser()

    return title_chain.invoke({
        "text": transcript[:4000]
    }).strip()



if __name__ == "__main__":

    sample_transcript = """
    Ahmad: Today we are discussing our AI Meeting Assistant.

    Ali: I will complete the frontend by Wednesday.

    Sara: I will handle the backend API using FastAPI.

    Ahmad: We decided to use PostgreSQL as our database.

    Ali: Should we add PDF export functionality?

    Ahmad: Yes, we will add PDF export in the next version.

    Sara: I will complete the backend by Thursday.
    """

    print("\nGENERATING MEETING TITLE...")

    title = generate_title(sample_transcript)

    print("Meeting Title:", title)

    print("\nGENERATING MEETING SUMMARY...")

    summary = summarize(sample_transcript)

    print("\nMEETING SUMMARY")
    print("=" * 50)
    print(summary)
