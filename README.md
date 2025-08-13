## About Dataset
The primary purpose of this dataset is to use it as a corpus for Retrieval Augmented Generation (RAG) with Large Language Models (LLM).

It has 2 query engines: with and without llamaindex

This dataset is a collection of text files of Amazon Web Services (AWS) case studies and blog articles related to Generative AI and Large Language Models.

90% of the text files are case studies and the rest are blog articles. This dataset was created by saving the HTML source of the AWS Case Study webpage and then passing the HTML file through a Python script that would extract the relevant text using BeautifulSoup Python library.

The cleaning process removes the HTML and tries only to keep the text context related to the case study, but it isn't an immaculate process.
