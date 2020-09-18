from allennlp.data import Vocabulary, DataLoader
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer

from build.lib.nlp_tools.nlp_models.multilayer_pretrained_transformer_mismatched_embedder import \
    MultilayerPretrainedTransformerMismatchedEmbedder
from nlp_tools.data_io.datasets import TokenizedSentencesDataset

if __name__ == "__main__":
    encoder_name = "bert-large-cased"
    print("loading indexer")
    indexer = PretrainedTransformerMismatchedIndexer(encoder_name)
    print("loading embedder")
    embedder = MultilayerPretrainedTransformerMismatchedEmbedder(encoder_name, layers_to_merge=[-1, -2, -3, -4],
                                                                 )

    s1 = "When the New York Stock Exchange heard the announcement , equities plummeted , causing a chain reaction of bank runs and failures throughout the United States that signaled the arrival of the Panic of 1873 to American shores .".split()
    s2 = "On the north bank of the river , WIS 42 turns northwest onto North Water Street and follows the river in a northerly direction to Forestville in Door County , where it becomes Forestville Avenue .".split()
    sentences = [s1, s2]
    dataset = TokenizedSentencesDataset(sentences, indexer)
    dataset.index_with(Vocabulary())
    # iterator = get_bucket_iterator(dataset, 1000, is_trainingset=False,
    #                                device=torch.device("cuda"))
    iterator = DataLoader(dataset, batch_size=2)
    outputs = list()
    import numpy as np
    for batch in iterator:
        batch_output = list()
        net_out = embedder(**batch["tokens"]["tokens"])
        a = net_out[0][18].detach().numpy()
        b = net_out[1][3].detach().numpy()
        a = a / np.linalg.norm(a, keepdims=True)
        b = b / np.linalg.norm(b, keepdims=True)
        score = np.matmul(a, b)

        print(score)