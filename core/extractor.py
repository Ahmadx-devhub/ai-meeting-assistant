
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

MODEL_NAME = "openai/gpt-oss-120b"


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

    return ChatGroq(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=2048,
        groq_api_key=api_key,
        timeout=120,
        max_retries=2,
    )


def split_transcript(transcript: str) -> list[str]:
    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=6000,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_text(transcript)


def reduce_results(results: list[str], llm, task: str) -> str:
    """Combine partial results in small batches."""

    while len(results) > 1:
        combined_results = []

        for i in range(0, len(results), 5):
            batch = results[i:i + 5]

            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    f"""
                    You are an expert meeting analyst.

                    Combine these partial results for: {task}

                    Rules:
                    - Remove duplicate information.
                    - Preserve important details.
                    - Never invent information.
                    - Keep names, dates, owners and deadlines.
                    - Do not lose important findings.
                    """
                ),
                ("human", "{text}"),
            ])

            chain = prompt | llm | StrOutputParser()

            output = chain.invoke({
                "text": "\n\n".join(batch)
            })

            combined_results.append(output)

        results = combined_results

    return results[0] if results else ""


def extract_with_map_reduce(
    transcript: str,
    instructions: str,
    task: str,
    empty_message: str,
) -> str:

    chunks = split_transcript(transcript)
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"""
            You are an expert meeting analyst.

            Task: {task}

            {instructions}

            Analyze ONLY the transcript section provided.

            Rules:
            - Do not invent information.
            - Preserve important details.
            - If this section contains no relevant information,
              return exactly: {empty_message}
            """
        ),
        ("human", "{text}"),
    ])

    map_chain = map_prompt | llm | StrOutputParser()

    partial_results = []

    for i, chunk in enumerate(chunks):
        print(f"Extracting {task}: chunk {i + 1}/{len(chunks)}")

        result = map_chain.invoke({"text": chunk})

        if result.strip() and result.strip() != empty_message:
            partial_results.append(result)

    if not partial_results:
        return empty_message

    return reduce_results(partial_results, llm, task)


def extract_action_items(transcript: str) -> str:
    return extract_with_map_reduce(
        transcript=transcript,
        task="Action Items",
        empty_message="No action items found.",
        instructions="""
        Extract action items.

        For each item provide:
        1. Task description
        2. Owner
        3. Deadline

        If owner or deadline is unknown, write "Not specified".
        Return a numbered list.
        """
    )


def extract_key_decisions(transcript: str) -> str:
    return extract_with_map_reduce(
        transcript=transcript,
        task="Key Decisions",
        empty_message="No key decisions found.",
        instructions="""
        Extract actual decisions made during the meeting.

        For each decision provide:
        1. Decision
        2. Reason
        3. People involved

        Do not confuse suggestions with final decisions.
        Return a numbered list.
        """
    )


def extract_questions(transcript: str) -> str:
    return extract_with_map_reduce(
        transcript=transcript,
        task="Questions",
        empty_message="No questions found.",
        instructions="""
        Extract important questions raised during the meeting.

        For each question provide:
        1. Question
        2. Person who asked it
        3. Answer or response, if available
        4. Status: Answered or Unanswered

        Return a numbered list.
        """
    )


def analyze_meeting(transcript: str) -> dict:
    if not transcript or not transcript.strip():
        raise ValueError("Meeting transcript cannot be empty.")

    return {
        "action_items": extract_action_items(transcript),
        "key_decisions": extract_key_decisions(transcript),
        "questions": extract_questions(transcript),
    }


if __name__ == "__main__":
    sample_transcript = """
    Ahmad: We need to launch our AI meeting assistant next Friday.
    Ali: I will complete the frontend by Wednesday.
    Sara: I will handle the backend API by Thursday.
    Ahmad: We decided to use FastAPI for the backend.
    Ali: Should we use PostgreSQL or SQLite?
    Sara: PostgreSQL would be better for production.
    Ahmad: Agreed. We will use PostgreSQL.
    """

    results = analyze_meeting(sample_transcript)

    for section, content in results.items():
        print(f"\n{'=' * 50}")
        print(section.upper())
        print(content)
