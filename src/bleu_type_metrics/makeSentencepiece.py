import os
import argparse
import sentencepiece as spm


def parserToken(add_help=True):
    desc = """
    inputs corpus, sentence
    output tokenized sentence or word embeddings or sentence emnbedding
    """
    parser = argparse.ArgumentParser(description=desc, add_help=add_help)
    spm = parser.add_argument_group("spm", "sentencepiece option")
    spm.add_argument("-train", "--train_file", type=str)
    spm.add_argument("-pretrained", "--pretrained",
                     action="store_true", default=False)
    spm.add_argument("-vocab_size", "--vocab_size", type=int, default=16000)
    spm.add_argument("-add_token", "--additional_tokens",
                     default=["<user_name>", "<url>"], nargs="*")
    spm.add_argument("-train_size", "--train_sentence_size",
                     type=int, default=0)
    spm.add_argument("-model_type", "--model_type", default="unigram",
                     choices=["word", "unigram", "char", "bpe"])
    spm.add_argument("-coverage", "--character_coverage",
                     type=float, default=0.9995)
    return parser


def sentencepiece(args, corpus):

    assert args.train_file, "input train files"
    if not args.pretrained:
        if os.path.isfile(args.train_file):
            args.train_file = os.path.dirname(args.train_file)
        write_tmp = os.path.join(args.train_file, "train_tmp.txt")
        print(f"write corpus in {write_tmp} to train")
        with open(write_tmp, "w") as f:
            for sent in corpus:
                f.write(sent+"\r\n")
        base_dir = os.getcwd()
        os.chdir(args.train_file)
        model_prefix = f"{args.model_type}_vocab{args.vocab_size}"
        additional_tokens = ",".join(args.additional_tokens)
        train_param = [f"--input={os.path.basename(write_tmp)}"]
        train_param.append(f"--model_prefix={model_prefix}")
        train_param.append(f"--vocab_size={args.vocab_size}")
        train_param.append(f"--user_defined_symbols={additional_tokens}")
        train_param.append(f"--model_type={args.model_type}")
        train_param.append(f"--character_coverage={args.character_coverage}")
        if args.train_sentence_size != 0:
            train_param.append(
                f"--input_sentence_size={args.train_sentence_size}")
            train_param.append(f"--shuffle_input_sentence=true")
        print("training...")
        spm.SentencePieceTrainer.Train(" ".join(train_param))
        os.chdir(base_dir)
        sp_model_path = os.path.join(args.train_file, model_prefix+".model")
        args.trainfile = sp_model_path
