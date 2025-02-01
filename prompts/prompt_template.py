doc_attr_prompt_self = '''Please summarize the attributes that make up the provided document, and generate a new question based on each attribute.
### Document:
{doc}
### important:
1. The new questions should be **unrelated** to the original document.
2. The new questions should align with the selected attribute and it is independent of other attributes.
3. The new questions should be independent of each other.
4. The response to the new question does not require information from the document.
5. Do not provide a solution or answer to the question.
Your output should be as follows:
### Attributes:
Here are the attributes.
### New Questions:
Here are the new questions.'''

doc_com_prompt_self = '''Please generate a new, high quality, reasonable and more challenging version of the given question.
### Important:
You only need to generate the new question, do not provide a solution or answer to the new question!
### Original Question:
{question}
Your output should be formatted as follows:
[New Question]: Here is the new question.'''