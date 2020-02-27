from nlp_models.huggingface_wrappers import GenericHuggingfaceWrapper, HuggingfaceModelNames
import numpy as np

from lxml import etree
root = etree.parse("/home/tommaso/Documents/data/WSD_Evaluation_Framework/Evaluation_Datasets/senseval2/senseval2.data.xml").getroot()
all_sentences = list()
for sentence_xml in root.findall("text/sentence"):
    sentence = list()
    for token in sentence_xml:
        sentence.append(token.text)
    all_sentences.append(sentence)

all_sentences = all_sentences[:100]

hf_model = GenericHuggingfaceWrapper(HuggingfaceModelNames.XLM_ROBERTA_LARGE.value, "cuda", token_limit=200)
hf_model.sentences_forward(np.array(all_sentences))