from nlp_models.huggingface_wrappers import GenericHuggingfaceWrapper

if __name__ == "__main__":
    model = GenericHuggingfaceWrapper("bert-base-cased", "cuda")
    out = model(["this is a test"])
    print(out)