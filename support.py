
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredURLLoader
from scrape import scrape_and_save_text
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter


#load the text from the url
url = 'https://www.mosaicml.com/blog/mpt-7b'
filename = 'output.txt'
filename = scrape_and_save_text(url, filename)
loader = UnstructuredFileLoader("output.txt")

data = loader.load()

print(data)




dummy_data = "Introducing MPT-7B: A New Standard for Open-Source, Commercially Usable LLMs | Databricks Blog\n\nURL: https://www.mosaicml.com/blog/mpt-7b\n\nScraped at: 2024-01-19 17:01:28.216991\n\nMay 5, 2023 in Generative AI\n\nIntroducing MPT-7B, the first entry in our MosaicML Foundation Series. MPT-7B is a transformer trained from scratch on 1T tokens of text and code. It is open source, available for commercial use, and matches the quality of LLaMA-7B. MPT-7B was trained on the MosaicML platform in 9.5 days with zero human intervention at a cost of ~$200k.\n\nLarge language models (LLMs) are changing the world, but for those outside well-resourced industry labs, it can be extremely difficult to train and deploy these models. This has led to a flurry of activity centered on open-source LLMs, such as the LLaMA series from Meta, the Pythia series from EleutherAI, the StableLM series from StabilityAI, and the OpenLLaMA model from Berkeley AI Research.\n\nToday, we at MosaicML are releasing a new model series called MPT (MosaicML Pretrained Transformer) to address the limitations of the above models and finally provide a commercially-usable, open-source model that matches (and - in many ways - surpasses) LLaMA-7B. Now you can train, finetune, and deploy your own private MPT models, either starting from one of our checkpoints or training from scratch. For inspiration, we are also releasing three finetuned models in addition to the b"

splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size=200,
    chunk_overlap=0
)


# chunks = splitter.split_text(dummy_data)

# we need something with many seperators, not just one

r_splitter = RecursiveCharacterTextSplitter(
    separators = ["\n\n", "\n", " "],  # List of separators based on requirement (defaults to ["\n\n", "\n", " "])
    chunk_size = 200,  # size of each chunk created
    chunk_overlap  = 0,  # size of  overlap between chunks in order to maintain the context
    length_function = len  # Function to calculate size, currently we are using "len" which denotes length of string however you can pass any token counter)
)

r_chunks = r_splitter.split_text(dummy_data)
print(len(r_chunks))
