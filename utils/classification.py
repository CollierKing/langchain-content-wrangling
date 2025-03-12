from typing import Optional, List
from colorama import Fore
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


# MARK: - HELPERS
def _format_examples(examples):
    formatted = ""
    for text, data in examples:
        data_json = data.json()
        formatted += f"Input: {text}\n"
        formatted += f"Extracted Data:\n```json\n{data_json}\n```\n\n"
    return formatted


def cls_color_mapping(cls: str):
    return {
        "strength": Fore.GREEN,
        "opportunity": Fore.LIGHTMAGENTA_EX,
        "milestone": Fore.LIGHTCYAN_EX,
        "weakness": Fore.RED,
        "challenge": Fore.YELLOW,
        "unclassified": Fore.WHITE
    }.get(cls)




# MARK: - SCHEMAS
class Note(BaseModel):
    """
    Schema for classification.
    """
    text: Optional[str] = \
        Field(
            None,
            description="The original text"
        )
    note: Optional[str] = \
        Field(
            None,
            description="The phrase highlighting a strength, opportunity, milestone, weakness, challenge"
        )
    classification: Optional[str] = \
        Field(
            None,
            description="One of 'strength, opportunity, milestone, weakness, challenge, unclassified'"
        )


class Data(BaseModel):
    """Extracted classifications."""
    results: List[Note]


# MARK: - FEW SHOT EXAMPLES
examples = [
    (
        "We saw strong revenue growth in the quarter.",
        Data(
            results=[
                Note(
                    text="We saw strong revenue growth in the quarter.",
                    note="strong revenue growth.",
                    classification="strength"
                )]),
    ),
    (
        "The quarter saw some deceleration in our product performance.",
        Data(
            results=[
                Note(
                    text="The quarter saw some deceleration in our product performance.",
                    note="deceleration in our product performance.",
                    classification="weakness"
                )]),
    ),
    (
        "We believe that this market will grow above expectations.",
        Data(
            results=[
                Note(
                    text="We believe that this market will grow above expectations.",
                    note="grow above expectations.",
                    classification="opportunity"
                )]),
    ),
    (
        "We saw record sales in the third quarter.",
        Data(
            results=[
                Note(
                    text="We saw record sales in the third quarter.",
                    note="record sales.",
                    classification="milestone"
                )]),
    ),
    (
        "We saw currency headwinds in the quarter.",
        Data(
            results=[
                Note(
                    text="We saw currency headwinds in the quarter.",
                    note="currency headwinds.",
                    classification="challenge"
                )]),
    ),
    (
        "We had $500M in revenue this quarter.",
        Data(
            results=[
                Note(
                    text="We had $500M in revenue this quarter.",
                    note=None,
                    classification="unclassified"
                )
            ]),
    ),
]

# MARK: - PROMPT
classification_prompt_template = \
    PromptTemplate(
        input_variables=["text"],
        partial_variables={"examples_text": _format_examples(examples)},
        template="""
            You are an expert sentiment classification algorithm. 
            For each provided text, return one classification of the following 
            (strength, opportunity, milestone, weakness, challenge, unclassified).
            
            Only extract relevant information containing notes from the text. 
            If you do not know the value of an attribute, return null for that attribute. 
            Do not guess.

            Here are some examples:
            {examples_text}
            Please provide the extracted data in JSON format matching the schema of the 'Note' class, 
            and enclose it within triple backticks. Do not include any additional text or explanations.

            Text: {text}
            """
    )
