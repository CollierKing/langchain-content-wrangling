from langchain_core.prompts import PromptTemplate

map_prompt = PromptTemplate.from_template("""
    The following is a document:
    {text}
    Please identify the main themes of this document. 
    Pay close attention to mentions of financial performance, 
    product strength/weakness, geographic strength/weakness, 
    future business outlook and financial guidance.
    No yapping, no preamble, no other remarks.
    """)

combine_prompt = PromptTemplate.from_template("""
    The following are summaries of different documents:
    {text}
    Please provide a consolidated summary of the main themes.
    Pay close attention to mentions of financial performance, 
    product strength/weakness, geographic strength/weakness, 
    future business outlook and financial guidance.
    No yapping, no preamble, no other remarks.
    """)