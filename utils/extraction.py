import json
import re

import pydantic
from colorama import Fore
from langchain_community.graphs.index_creator import GraphIndexCreator, KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT
from typing import List, Optional

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from .classification import Data, _format_examples


# MARK: - HELPERS
# MARK: - Graphs
async def agraph_inference(ic: GraphIndexCreator, text: str, custom_prompt: str = None):
    """

    :param ic:
    :param text:
    :param custom_prompt:
    :return:
    """
    return await ic.afrom_text(
        text=text,
        prompt=custom_prompt or KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT
    )


# MARK: - TEXT
def parse_output(output: str, schema: any):
    """
    Parse the output of extraction into JSON

    :param schema:
    :param output:
    :return:
    """
    try:
        # Extract JSON from within triple backticks
        pattern = r'```json\n(.*?)```'
        match = re.search(pattern, output, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            data = schema.parse_raw(json_str)
        elif re.search("```", output, re.DOTALL):
            data = json.loads(output.replace("```", ""))
        else:
            # Attempt to parse the entire output
            data = schema.parse_raw(output)
    except Exception as e:
        print(f"Error parsing output: {e}")
        data = schema(results=[])
    return data


def split_sentences(text: str):
    return re.split(r'\.\s+(?=[A-Z])', text)


def get_statistic(text: str):
    number_word_list = [
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
        'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
        'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty',
        'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'hundreds', 'thousand',
        'thousands', 'million', 'millions', 'billion', 'billions', 'trillion', 'trillions'
    ]

    # Build a pattern like: \bone\b|\btwo\b|...|\btrillions\b
    number_words_pattern = r'|'.join(fr'\b{word}\b' for word in number_word_list)

    # Combine it with your numeric pattern:
    numeric_pattern = r'\b\d+(\.\d+)?%?\b'
    pattern = re.compile(rf'{numeric_pattern}|{number_words_pattern}', re.IGNORECASE)

    # If we find a match, return the entire text
    if pattern.search(text):
        return text
    return None


def build_structured_results(
        text: str,
        extracted_data: any,
        arr_results: list,
        arr_not_matched: list,
        schema: any,
):
    try:
        # Validate input otherwise skip
        if type(extracted_data) == dict:
            arr_og_text = extracted_data.get("results")
            if len(arr_og_text) == 0:
                return

            extracted_data = \
                Data(
                    results=[
                        schema(
                            text=x.get("text"),
                            note=x.get("note"),
                            classification=x.get("classification"),
                        )
                        for x in arr_og_text
                    ]
                )

        # Check that the return string matches the input string
        # Otherwise could have hallucinated...so skip it
        og_text = extracted_data.results[0].text if len(extracted_data.results) > 0 else ""
        og_text = og_text.rstrip(".")
        if text.find(og_text) >= 0:
            arr_results.append(extracted_data)
        else:
            print(Fore.BLUE + "Inference string could not be matched...")
            arr_not_matched.append(extracted_data)

    except Exception as e:
        print(Fore.LIGHTRED_EX + "Skipping..." + str(e))
        print(Fore.LIGHTRED_EX + text)
        return


# MARK: - SCHEMAS
class Stat(BaseModel):
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
            description="The phrase highlighting the statistics, numbers, figures or percentages"
        )
    classification: Optional[str] = \
        Field(
            None,
            description="One of 'stat', 'not stat'"
        )


class Stats(BaseModel):
    """Extracted classifications."""
    results: List[Stat]


# MARK: - FEW SHOT EXAMPLES
examples = [
    (
        "There are 300 passengers on the airplane.",
        Stats(
            results=[
                Stat(
                    text="There are 300 passengers on the airplane.",
                    note="300 passengers.",
                    classification="stat"
                )]),
    ),
    (
        "The date is March 11, 2025.",
        Stats(
            results=[
                Stat(
                    text="The date is March 11, 2025.",
                    note=None,
                    classification="not stat"
                )]),
    ),
    (
        "One-fifth of the students got A's on the test",
        Stats(
            results=[
                Stat(
                    text="One-fifth of the students got A's on the test",
                    note="One-fifth of the students.",
                    classification="stat"
                )]),
    ),
    (
        "I have three things on my mind.",
        Stats(
            results=[
                Stat(
                    text="I have three things on my mind.",
                    note=None,
                    classification="not stat"
                )]),
    ),
    (
        "My property taxes went up by 9.99%.",
        Stats(
            results=[
                Stat(
                    text="My property taxes went up by 9.99%.",
                    note="property taxes went up by 9.99%.",
                    classification="stat"
                )]),
    ),
    (
        "The document number is A1024.",
        Stats(
            results=[
                Stat(
                    text="The document number is A1024.",
                    note=None,
                    classification="not stat"
                )]),
    )
]

# MARK: - PROMPT
extraction_prompt_template = \
    PromptTemplate(
        input_variables=["text"],
        partial_variables={"examples_text": _format_examples(examples)},
        template="""
            You are an expert statistic extraction algorithm. 
            For each provided text, return one classification of the following 
            (stat, not stat).
            
            Only extract relevant information containing statistics, numbers, figures or percentages from the text. 
            If you do not know the value of an attribute, return null for that attribute. 
            Do not guess.

            Here are some examples:
            {examples_text}
            Please provide the extracted data in JSON format matching the schema of the 'Stat' class, 
            and enclose it within triple backticks. Do not include any additional text or explanations.

            Text: {text}
            """
    )
