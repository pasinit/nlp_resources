import unittest
from unittest import TestCase

import torch
import numpy as np
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from nlp_models.bert_wrappers import BertSentencePredictionWrapper, BertTokeniserWrapper, BertWrapper, BertNames


class BertTester(TestCase):

    def test_bert_sentence_prediction_output(self):
        bert_model = BertSentencePredictionWrapper("bert-base-multilingual-cased", "cuda")
        out, *_ = bert_model([("this is a sentence.", "the computer are nice persons!")])
        self.assertEqual(out.shape, torch.Size([1, 2]))
        cls = torch.argmax(out, -1)[0].item()
        self.assertEqual(cls, 1)
        print(out.tolist())

        out, *_ = bert_model([("this is a sentence", "and this is its consequence")])
        self.assertEqual(out.shape, torch.Size([1, 2]))
        cls = torch.argmax(out, -1)[0].item()
        self.assertEqual(cls, 0)
        print(out.tolist())

    def test_bert_tokeniser(self):
        test_sentence = "this is a stupid test for the tokeniser"
        tokeniser = BertTokeniserWrapper("bert-base-multilingual-cased", "cuda")

        tokens, indexed_tokens, segment_ids, attention_masks = tokeniser.tokenise([test_sentence])
        correct_token_class = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.assertEqual(tokens[0][0], "[CLS]")
        self.assertTrue(len([x for x in tokens[0] if x == "[SEP]"]) == 1)
        self.assertEqual(segment_ids.tolist()[0], correct_token_class)

        test_pair = (test_sentence, "another stupid sentence")

        tokens, indexed_tokens, segment_ids, attention_masks = tokeniser.tokenise([test_pair])
        correct_token_class = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

        self.assertEqual(tokens[0][0], "[CLS]")
        self.assertTrue(len([x for x in tokens[0] if x == "[SEP]"]) == 2)
        self.assertEqual(segment_ids.tolist()[0], correct_token_class)
        # TODO test batched sentences and limit cases (empty strings) and test strings longer than max tok size.
        # print(tokens)
        # print(indexed_tokens.tolist())
        # print(segment_ids.tolist())
        # print(attention_masks.tolist())

    def test_word_merging(self):
        bert = BertWrapper(BertNames.BERT_LARGE_CASED_WHOLE_WORD_MASKING.value, "cuda", token_limit=100)
        test_sentence = ["this is a stupid test for the tokeniser".split(" "),
                         "this is a second stupid test for the tokeniser".split(" ")]
        with torch.no_grad():
            outputs, bert_in = bert.word_forward(np.array(test_sentence))
        hidde_states = outputs["hidden_states"]
        assert max([len(x) for x in test_sentence]) == hidde_states.shape[1]
        assert len(test_sentence) == hidde_states.shape[0]

        listoftests = ["this is a second stupid test for the tokeniser".split(" "),
                       "this is a second stupid test for the tokeniser".split(" ")] * 100
        with torch.no_grad():
            outputs, bert_in = bert.word_forward(np.array(listoftests))
        hidde_states = outputs["hidden_states"]
        assert max([len(x) for x in test_sentence]) == hidde_states.shape[1]
        assert len(listoftests) == hidde_states.shape[0] or print(len(listoftests), hidde_states.shape[0])
        listoftests = [
            "Marqu ( དམར་, 马乡 ) is a township in Doilungdêqên District in the Tibet Autonomous Region of China .".split(),
            "Majee holds two patents , and has published a number of articles, ResearchGate, an online repository of scientific articles has listed 49 of them .".split(),
            "On November 12 , 2008 , Super Shuffle was removed from the online line-up when the \" merged \" lineups were put in place between XM and Sirius .".split(),
            "\" Numb \" is a song by Canadian musician Holly McNarland , released as the first single from her debut studio album , Stuff .".split(),
            "Lugs are the loops ( or protuberances ) that exist on both arms of a hinge , featuring a hole for the axis of the hinge .".split(),
            "In September 1885 , the then-unincorporated area was the scene of an attack on Chinese laborers who had come to pick hops from local fields .".split(),
            "Three of the laborers died from gunshot wounds , and none of the attackers were convicted of any wrongdoing .".split()]
        with torch.no_grad():
            outputs, bert_in = bert.word_forward(np.array(listoftests))
        hidde_states = outputs["hidden_states"]
        assert max([len(x) for x in listoftests]) == hidde_states.shape[1]
        assert len(listoftests) == hidde_states.shape[0] or print(len(listoftests), hidde_states.shape[0])

    def test_semantic_correctness_word_merging(self):
        bert_model_name = BertNames.BERT_BASE_MULTILINGUAL_CASED.value
        test_s1 = "Central bank services can also foster European financial integration ."
        test_s2 = "Guidance has a functional interface with the Federation memory bank ."
        test_s3 = "Our clan left my uncle and his wife by the river bank to have their honeymoon ."

        bert_tokeniser = BertTokenizer.from_pretrained(bert_model_name)

        bert_vanilla = BertModel.from_pretrained(bert_model_name)
        test_s1_b = bert_tokeniser.tokenize("[CLS] " + test_s1 + " [SEP]")  # bank is at 2
        test_s2_b = bert_tokeniser.tokenize("[CLS] " + test_s2 + " [SEP]")  # bank is at 12
        test_s3_b = bert_tokeniser.tokenize("[CLS] " + test_s3 + " [SEP]")  # bank is at 12
        test_s1_tok = bert_tokeniser.convert_tokens_to_ids(test_s1_b)
        test_s2_tok = bert_tokeniser.convert_tokens_to_ids(test_s2_b)
        test_s3_tok = bert_tokeniser.convert_tokens_to_ids(test_s3_b)
        vanilla_out_1 = bert_vanilla(torch.LongTensor(test_s1_tok).unsqueeze(0))[0]
        vanilla_out_2 = bert_vanilla(torch.LongTensor(test_s2_tok).unsqueeze(0))[0]
        vanilla_out_3 = bert_vanilla(torch.LongTensor(test_s3_tok).unsqueeze(0))[0]
        vanilla_bank_1 = vanilla_out_1[0][2]
        vanilla_bank_2 = vanilla_out_2[0][12]
        vanilla_bank_3 = vanilla_out_3[0][12]

        bert = BertWrapper(bert_model_name, "cpu", token_limit=100)
        with torch.no_grad():
            outputs, bert_in = bert.word_forward(np.array([test_s1.split(), test_s2.split(), test_s3.split()]))
        hidden_states = outputs["hidden_states"]
        bank1 = hidden_states[0][1]
        bank2 = hidden_states[1][9]
        bank3 = hidden_states[2][11]
        cossim = torch.nn.CosineSimilarity(-1)
        sim1 = cossim(bank1, vanilla_bank_1).item()
        sim2 = cossim(bank2, vanilla_bank_2).item()
        sim3 = cossim(bank3, vanilla_bank_3).item()

        self.assertAlmostEqual(sim1, 1.0, delta=1.0e-6, msg=sim1-1.0)
        self.assertAlmostEqual(sim2, 1.0, delta=1.0e-6, msg=sim2-1.0)
        self.assertAlmostEqual(sim3, 1.0, delta=1.0e-6, msg=sim3-1.0)

        self.assertGreater(cossim(bank1, bank2), cossim(bank1, bank3))
        self.assertGreater(cossim(bank2, bank1), cossim(bank2, bank3))

        test_s1 = "A mouse , plural mice , is a small rodent characteristically having a pointed snout ." #animal
        test_s2 = "An optical mouse is a computer mouse which uses a light source , typically a light-emitting diode ( LED ) , and a light detector ." #pc
        test_s3 = "A bus mouse is a variety of PC computer mouse which is attached to the computer using ." #pc

        with torch.no_grad():
            outputs, bert_in = bert.word_forward(np.array([test_s1.split(), test_s2.split(), test_s3.split()]))
        hidden_states = outputs["hidden_states"]
        mouse1 = hidden_states[0][1]
        mouse2 = hidden_states[1][6]
        mouse3 = hidden_states[2][2]
        cossim = torch.nn.CosineSimilarity(-1)
        self.assertGreater(cossim(mouse2, mouse3), cossim(mouse2, mouse1))
        self.assertGreater(cossim(mouse2, mouse3), cossim(mouse3, mouse1))

        test_s1 = "A balance spring , or hairspring , is a spring attached to the balance wheel in mechanical timepieces ." #metal device
        # test_s2 = "A rhythmic spring is a cold water spring from which the flow of water either varies or starts and stops entirely ." #water spring #idx 7
        test_s2 = "Meteorologists generally define four seasons in many climatic areas : spring , summer , autumn ( fall ) and winter ."
        test_s3 = "The portion of the spring between the stud and the slot is held stationary , so the position of the slot controls the free length of the spring ." #metal device

        with torch.no_grad():
            outputs, bert_in = bert.word_forward(np.array([test_s1.split(), test_s2.split(), test_s3.split()]))
        hidden_states = outputs["hidden_states"]
        spring2 = hidden_states[1][10]
        spring1 = hidden_states[0][2]
        spring3 = hidden_states[2][4]
        cossim = torch.nn.CosineSimilarity(-1)
        self.assertGreater(cossim(spring1, spring3), cossim(spring2, spring3))
        self.assertGreater(cossim(spring1, spring3), cossim(spring1, spring2))




    def test_memory_bert(self):
        bert = BertWrapper(BertNames.BERT_BASE_MULTILINGUAL_CASED.value, "cuda", token_limit=512)
        bert.eval()
        s = " ".join(["cane"] * 298)
        s = [s] * 200
        print(np.array(s).shape)
        for i in tqdm(range(10)):
            out = bert(s)


# if __name__ == "__main__":
#     BertTester().test_bert_sentence_prediction_output()
if __name__ == '__main__':
    # BertTester().test_word_merging()
    BertTester().test_semantic_correctness_word_merging()
    # BertTester().test_memory_bert()
    # unittest.main()
