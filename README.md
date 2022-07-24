# Instituto Atlântico's Cognitive Computing Bootcamp

This repository is comprised of several assigments from [Instituto Atlântico's Cognitive Computing Bootcamp](https://www.atlantico.com.br/academy-bootcamp/). Topics covered are machine learning, natural language processing and digital image processing.

Most of the coding is done using Python 3.

In the following sections there is a short introduction about each project.

---

## 1. Identifying Relevant Terms in Medical Records Through Natural Language Processing

This project provides automated identification of relevant terms in medical records by use of classical NLP algorithms.

Input texts are written in Portuguese.

### How to run this project

1. Create a virtual environment:

`python3 -m venv env`

2. Activate it:

`source env/bin/activate`

3. Install the dependencies:

`pip install -r requirements.txt`

4. Install additional packages, if needed

> Nltk and stanza libraries have additional packages that have to be downloaded

`python3 src/nlp/utils/downloads.py`


5. Run the application entry point:

`python3 app_nlp_project.py`

---

## 2. Extracting Features from Images of Leaves

This project aims to automate feature extraction from images of leaves, such as dimensions and area.

All images are captured with a standard resolution of 8MP, and three segmentation algoritms are applied and have results confronted among them.

1. Create a virtual environment:

`python3 -m venv env`

2. Activate it:

`source env/bin/activate`

3. Install the dependencies:

`pip install -r requirements.txt`

4. Install additional packages, if needed

TODO no additional packages for now


5. Run the application entry point:

`python3 app_pdi_project.py`


# Team (**Squad 1**)

| Team member                | Github        | Phone/Whats     |
| -------------------------- | ------------- | --------------- |
| Edson                      |               |                 |
| Emerson                    |               |                 |
| Felipe Araújo              | felipe-araujo |                 |
| Paulo Vinicius P. Pinheiro | paulovpp      | (88)9 9723-6607 |
| Rômulo                     | romulopm2     |                 |


Project template: https://github.com/Alysonbnr/template_bootcamp