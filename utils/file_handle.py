import sys, fitz

sys.stdout.reconfigure(encoding='utf-8')


def read_file(file_path):
    """
    This function reads a file and returns its content as a string.
    Author: Paulo Vinicius P. Pinheiro
    Date: April, 2022
    Ver: 1.0

    :param file_path: string with the path to the document.
    :return: the document in string type.

    """

    doc = fitz.open(file_path)  # open document
    for page in doc: 
        text = page.get_text().encode("utf8")  # get plain text (is in UTF-8)

    text_str = text.decode("utf8")
    return text_str


def generate_csv(file, database):
    """
    Generate a CSV file with all the words from the documents and the respective nearest words and its weights

    :param file: The path to save the CSV.
    :param database: The dataframe with the data.
    :return: If path_or_buf is None, returns the resulting csv format as a string. Otherwise returns None.
    """
    df = database
    return df.to_csv(file, index = False)
