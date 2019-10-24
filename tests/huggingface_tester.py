import unittest
from unittest import TestCase

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from nlp_models.huggingface_wrappers import GenericHuggingfaceWrapper, HuggingfaceModelNames
from utils.huggingface_utils import get_needed_start_end_sentence_tokens, get_tokenizer_kwargs

from utils.huggingface_utils import get_model_kwargs


class HuggingfaceTester(TestCase):

    def test_huggingface_models_word_merging(self):
        for model_name in HuggingfaceModelNames:
            print(model_name.value)
            self.test_word_merging(GenericHuggingfaceWrapper(model_name.value, "cuda"))

    # def test_word_merging(self, model: GenericHuggingfaceWrapper):
    #
    #     test_sentence = ["this is a stupid test for the tokeniser".split(" "),
    #                      "this is a second stupid test for the tokeniser".split(" ")]
    #     with torch.no_grad():
    #         outputs, bert_in = model.word_forward(np.array(test_sentence))
    #     hidde_states = outputs["hidden_states"]
    #     assert max([len(x) for x in test_sentence]) == hidde_states.shape[1]
    #     assert len(test_sentence) == hidde_states.shape[0]
    #
    #     listoftests = ["this is a second stupid test for the tokeniser".split(" "),
    #                    "this is a second stupid test for the tokeniser".split(" ")] * 100
    #     with torch.no_grad():
    #         outputs, bert_in = model.word_forward(np.array(listoftests))
    #     hidde_states = outputs["hidden_states"]
    #     assert max([len(x) for x in test_sentence]) == hidde_states.shape[1]
    #     assert len(listoftests) == hidde_states.shape[0] or print(len(listoftests), hidde_states.shape[0])
    #     listoftests = [
    #         "Marqu ( དམར་, 马乡 ) is a township in Doilungdêqên District in the Tibet Autonomous Region of China .".split(),
    #         "Majee holds two patents , and has published a number of articles, ResearchGate, an online repository of scientific articles has listed 49 of them .".split(),
    #         "On November 12 , 2008 , Super Shuffle was removed from the online line-up when the \" merged \" lineups were put in place between XM and Sirius .".split(),
    #         "\" Numb \" is a song by Canadian musician Holly McNarland , released as the first single from her debut studio album , Stuff .".split(),
    #         "Lugs are the loops ( or protuberances ) that exist on both arms of a hinge , featuring a hole for the axis of the hinge .".split(),
    #         "In September 1885 , the then-unincorporated area was the scene of an attack on Chinese laborers who had come to pick hops from local fields .".split(),
    #         "Three of the laborers died from gunshot wounds , and none of the attackers were convicted of any wrongdoing .".split()]
    #     with torch.no_grad():
    #         outputs, bert_in = model.word_forward(np.array(listoftests))
    #     hidde_states = outputs["hidden_states"]
    #     assert max([len(x) for x in listoftests]) == hidde_states.shape[1]
    #     assert len(listoftests) == hidde_states.shape[0] or print(len(listoftests), hidde_states.shape[0])

    def test_huggingface_models_correctness_word_merging(self):
        for model_name in HuggingfaceModelNames:
            print(model_name.value)
            self.test_word_merging(model_name.value)

    def test_word_merging(self, model_name):
        test_s1 = "Central bank services can also foster European financial integration ."
        test_s2 = "Guidance has a functional interface with the Federation memory bank ."
        test_s3 = "Our clan left my uncle and his wife by the river bank to have their honeymoon ."

        vanilla_tokeniser = AutoTokenizer.from_pretrained(model_name)
        kwargs = get_tokenizer_kwargs(model_name)
        vanilla_model = AutoModel.from_pretrained(model_name)
        start, end = get_needed_start_end_sentence_tokens(model_name, vanilla_tokeniser)

        test_s1_b = vanilla_tokeniser.tokenize(
            ((start + " ") if start else "") + test_s1 + ((" " + end) if end else ""), **kwargs)
        test_s2_b = vanilla_tokeniser.tokenize(
            ((start + " ") if start else "") + test_s2 + ((" " + end) if end else ""), **kwargs)
        test_s3_b = vanilla_tokeniser.tokenize(
            ((start + " ") if start else "") + test_s3 + ((" " + end) if end else ""), **kwargs)
        tokeniser_kwargs = get_tokenizer_kwargs(model_name)
        test_s1_tok = vanilla_tokeniser.encode(test_s1_b, **tokeniser_kwargs)
        test_s2_tok = vanilla_tokeniser.encode(test_s2_b, **tokeniser_kwargs)
        test_s3_tok = vanilla_tokeniser.encode(test_s3_b, **tokeniser_kwargs)
        kwargs = get_model_kwargs(model_name, "cpu", {}, [0]*len(test_s1_tok), [1] * len(test_s1_tok))
        vanilla_out_1 = vanilla_model(torch.LongTensor(test_s1_tok).unsqueeze(0), **kwargs)[0]
        # token_type_ids=torch.LongTensor(token_type_ids_s1).unsqueeze(0))[0]
        kwargs = get_model_kwargs(model_name, "cpu", {}, [0]*len(test_s2_tok), [1] * len(test_s2_tok))
        vanilla_out_2 = vanilla_model(torch.LongTensor(test_s2_tok).unsqueeze(0), **kwargs)[0]
        # token_type_ids=torch.LongTensor(token_type_ids_s2).unsqueeze(0))[0]
        kwargs = get_model_kwargs(model_name, "cpu", {}, [0]*len(test_s3_tok), [1] * len(test_s3_tok))
        vanilla_out_3 = vanilla_model(torch.LongTensor(test_s3_tok).unsqueeze(0), **kwargs)[0]
        # token_type_ids=torch.LongTensor(token_type_ids_s3).unsqueeze(0))[0]
        index1 = test_s1_b.index("bank") if "bank" in test_s1_b else test_s1_b.index(
            "Ġbank") if "Ġbank" in test_s1_b else test_s1_b.index("▁bank")
        index2 = test_s2_b.index("bank") if "bank" in test_s2_b else test_s2_b.index(
            "Ġbank") if "Ġbank" in test_s2_b else test_s2_b.index("▁bank")
        index3 = test_s3_b.index("bank") if "bank" in test_s3_b else test_s3_b.index(
            "Ġbank") if "Ġbank" in test_s3_b else test_s3_b.index("▁bank")

        vanilla_bank_1 = vanilla_out_1[0][index1]
        vanilla_bank_2 = vanilla_out_2[0][index2]
        vanilla_bank_3 = vanilla_out_3[0][index3]

        hf_model = GenericHuggingfaceWrapper(model_name, "cpu", token_limit=100)
        with torch.no_grad():
            outputs, bert_in = hf_model.sentences_forward(np.array([test_s1.split(), test_s2.split(), test_s3.split()]))
        hidden_states = outputs["hidden_states"]
        bank1 = hidden_states[0][1]
        bank2 = hidden_states[1][9]
        bank3 = hidden_states[2][11]
        cossim = torch.nn.CosineSimilarity(-1)
        sim1 = cossim(bank1, vanilla_bank_1).item()
        sim2 = cossim(bank2, vanilla_bank_2).item()
        sim3 = cossim(bank3, vanilla_bank_3).item()

        self.assertAlmostEqual(sim1, 1.0, delta=1.0e-1, msg=1.0 - sim1)
        self.assertAlmostEqual(sim2, 1.0, delta=1.0e-1, msg=1.0 - sim2)
        self.assertAlmostEqual(sim3, 1.0, delta=1.0e-1, msg=1.0 - sim3)
        print(sim1, sim2, sim3)

    def test_semantic_correctness_word_merging(self, model_name):
        test_s1 = "Central bank services can also foster European financial integration ."
        test_s2 = "Guidance has a functional interface with the Federation memory bank ."
        test_s3 = "Our clan left my uncle and his wife by the river bank to have their honeymoon ."

        vanilla_tokeniser = AutoTokenizer.from_pretrained(model_name)
        kwargs = get_tokenizer_kwargs(model_name)
        vanilla_model = AutoModel.from_pretrained(model_name)
        start, end = get_needed_start_end_sentence_tokens(model_name, vanilla_tokeniser)

        test_s1_b = vanilla_tokeniser.tokenize(
            ((start + " ") if start else "") + test_s1 + ((" " + end) if end else ""), **kwargs)
        test_s2_b = vanilla_tokeniser.tokenize(
            ((start + " ") if start else "") + test_s2 + ((" " + end) if end else ""), **kwargs)
        test_s3_b = vanilla_tokeniser.tokenize(
            ((start + " ") if start else "") + test_s3 + ((" " + end) if end else ""), **kwargs)
        encoding_s1 = vanilla_tokeniser.encode_plus(test_s1_b)
        encoding_s2 = vanilla_tokeniser.encode_plus(test_s2_b)
        encoding_s3 = vanilla_tokeniser.encode_plus(test_s3_b)
        test_s1_tok, token_type_ids_s1 = [encoding_s1[x] for x in encoding_s1.keys()]
        test_s2_tok, token_type_ids_s2 = [encoding_s2[x] for x in encoding_s2.keys()]
        test_s3_tok, token_type_ids_s3 = [encoding_s3[x] for x in encoding_s3.keys()]
        vanilla_out_1 = vanilla_model(torch.LongTensor(test_s1_tok).unsqueeze(0),
                                      token_type_ids=torch.LongTensor(token_type_ids_s1).unsqueeze(0))[0]
        vanilla_out_2 = vanilla_model(torch.LongTensor(test_s2_tok).unsqueeze(0),
                                      token_type_ids=torch.LongTensor(token_type_ids_s2).unsqueeze(0))[0]
        vanilla_out_3 = vanilla_model(torch.LongTensor(test_s3_tok).unsqueeze(0),
                                      token_type_ids=torch.LongTensor(token_type_ids_s3).unsqueeze(0))[0]
        index1 = test_s1_b.index("bank") if "bank" in test_s1_b else test_s1_b.index(
            "Ġbank") if "Ġbank" in test_s1_b else test_s1_b.index("_bank")
        index2 = test_s2_b.index("bank") if "bank" in test_s2_b else test_s2_b.index(
            "Ġbank") if "Ġbank" in test_s2_b else test_s2_b.index("_bank")
        index3 = test_s3_b.index("bank") if "bank" in test_s3_b else test_s3_b.index(
            "Ġbank") if "Ġbank" in test_s3_b else test_s3_b.index("_bank")

        vanilla_bank_1 = vanilla_out_1[0][index1]
        vanilla_bank_2 = vanilla_out_2[0][index2]
        vanilla_bank_3 = vanilla_out_3[0][index3]

        hf_model = GenericHuggingfaceWrapper(model_name, "cpu", token_limit=100)
        with torch.no_grad():
            outputs, bert_in = hf_model.sentences_forward(np.array([test_s1.split(), test_s2.split(), test_s3.split()]))
        hidden_states = outputs["hidden_states"]
        bank1 = hidden_states[0][1]
        bank2 = hidden_states[1][9]
        bank3 = hidden_states[2][11]
        cossim = torch.nn.CosineSimilarity(-1)
        sim1 = cossim(bank1, vanilla_bank_1).item()
        sim2 = cossim(bank2, vanilla_bank_2).item()
        sim3 = cossim(bank3, vanilla_bank_3).item()

        self.assertAlmostEqual(sim1, 1.0, delta=1.0e-2, msg=sim1 - 1.0)
        self.assertAlmostEqual(sim2, 1.0, delta=1.0e-2, msg=sim2 - 1.0)
        self.assertAlmostEqual(sim3, 1.0, delta=1.0e-2, msg=sim3 - 1.0)

        self.assertGreater(cossim(bank1, bank2), cossim(bank1, bank3))
        self.assertGreater(cossim(bank2, bank1), cossim(bank2, bank3))

        test_s1 = "A mouse , plural mice , is a small rodent characteristically having a pointed snout ."  # animal
        test_s2 = "An optical mouse is a computer mouse which uses a light source , typically a light-emitting diode ( LED ) , and a light detector ."  # pc
        test_s3 = "A bus mouse is a variety of PC computer mouse which is attached to the computer using ."  # pc

        with torch.no_grad():
            outputs, bert_in = hf_model.word_forward(np.array([test_s1.split(), test_s2.split(), test_s3.split()]))
        hidden_states = outputs["hidden_states"]
        mouse1 = hidden_states[0][1]
        mouse2 = hidden_states[1][6]
        mouse3 = hidden_states[2][2]
        cossim = torch.nn.CosineSimilarity(-1)
        self.assertGreater(cossim(mouse2, mouse3), cossim(mouse2, mouse1))
        self.assertGreater(cossim(mouse2, mouse3), cossim(mouse3, mouse1))

        test_s1 = "A balance spring , or hairspring , is a spring attached to the balance wheel in mechanical timepieces ."  # metal device
        # test_s2 = "A rhythmic spring is a cold water spring from which the flow of water either varies or starts and stops entirely ." #water spring #idx 7
        test_s2 = "Meteorologists generally define four seasons in many climatic areas : spring , summer , autumn ( fall ) and winter ."
        test_s3 = "The portion of the spring between the stud and the slot is held stationary , so the position of the slot controls the free length of the spring ."  # metal device

        with torch.no_grad():
            outputs, bert_in = hf_model.word_forward(np.array([test_s1.split(), test_s2.split(), test_s3.split()]))
        hidden_states = outputs["hidden_states"]
        spring2 = hidden_states[1][10]
        spring1 = hidden_states[0][2]
        spring3 = hidden_states[2][4]
        cossim = torch.nn.CosineSimilarity(-1)
        self.assertGreater(cossim(spring1, spring3), cossim(spring2, spring3))
        self.assertGreater(cossim(spring1, spring3), cossim(spring1, spring2))

    # def test_memory_bert(self):
    #     bert = BertWrapper(BertNames.BERT_BASE_MULTILINGUAL_CASED.value, "cuda", token_limit=512)
    #     bert.eval()
    #     s = " ".join(["cane"] * 298)
    #     s = [s] * 200
    #     print(np.array(s).shape)
    #     for i in tqdm(range(10)):
    #         out = bert(s)


# if __name__ == "__main__":
#     BertTester().test_bert_sentence_prediction_output()
if __name__ == '__main__':
    # BertTester().test_word_merging()
    HuggingfaceTester().test_huggingface_models_correctness_word_merging()
    # BertTester().test_memory_bert()
    # unittest.main()
