import openai
def generate_text_embeddings(text):
    """
    Converts text to embeddings.

    Parameters:
    - text (str): The text to be converted into embeddings.

    Returns:
    - list: A list of embeddings (dummy data in this example).
    """
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )

    return response.data[0].embedding
