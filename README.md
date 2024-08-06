# Huggingface transformers

Docker images for [Huggingface transformers](https://github.com/huggingface/transformers).

Available versions:

* 4.43.1 ([CUDA 12.1](4.43.1_cuda12.1), [CUDA 12.1 langchain](4.43.1_cuda12.1_langchain), [CUDA 12.1 langchain/optional RAG](4.43.1-2_cuda12.1_langchain), [CUDA 12.1 MMS](4.43.1_cuda12.1_mms))
* 4.42.4-post ([CUDA 12.1](4.42.4-post_cuda12.1), [CUDA 12.1 langchain](4.42.4-post_cuda12.1_langchain))
* 4.42.3 ([CUDA 12.1](4.42.3_cuda12.1), [CUDA 12.1 langchain](4.42.3_cuda12.1_langchain))
* 4.40.2 ([CUDA 12.1](4.40.2_cuda12.1), [CUDA 12.1 langchain](4.40.2_cuda12.1_langchain))
* 4.40.0 ([CUDA 11.7](4.40.0_cuda11.7), [CUDA 11.7 langchain](4.40.0_cuda11.7_langchain))
* 4.36.0 ([CUDA 11.7](4.36.0_cuda11.7), [CUDA 11.7 Mistral](4.36.0_cuda11.7_mistral), [CUDA 11.7 Text classification](4.36.0_cuda11.7_classification))
* 4.35.0 ([CUDA 11.7](4.35.0_cuda11.7), [CUDA 11.7 Mistral](4.35.0_cuda11.7_mistral))
* 4.31.0 ([CUDA 11.7](4.31.0_cuda11.7), [CUDA 11.7 with falcontune](4.31.0_cuda11.7_falcontune_20230618), [CUDA 11.7 Llama2](4.31.0_cuda11.7_llama2) ([8bit](4.31.0_cuda11.7_llama2_8bit)), [CUDA 11.7 QLoRA](4.31.0_cuda11.7_qlora_20230724), [CUDA 11.7 RemBERT](4.31.0_cuda11.7_rembert), [CUDA 11.7 translate](4.31.0_cuda11.7_translate))
* 4.7.0 ([CUDA 11.1](4.7.0_cuda11.1), [CUDA 11.1 with finetune-gpt2xl](4.7.0_cuda11.1_finetune-gpt2xl_20220924))


## Restricted access

In case models or datasets require being logged into Huggingface, you can give your 
Docker container access via an access token.

### Create access token

In order to create an access token, do the following:
- Log into https://huggingface.co
- Go to *Settings* -> *Access tokens*
- Create a token (*read* access is sufficient, unless you want to push models back to huggingface)
- Copy the token onto the clipboard
- Save the token in a [.env file](https://hexdocs.pm/dotenvy/0.2.0/dotenv-file-format.html), using `HF_TOKEN` as the variable name

### Provide token to container

Add the following parameter to make all the environment variables stored in the `.env` file in 
the current directory available to your Docker container:

```
--env-file=`pwd`/.env
```

### Log into Huggingface

With the `HF_TOKEN` environment variable set, you can now log into Huggingface inside your Docker 
container using the following command:

```
huggingface-cli login --token=$HF_TOKEN
```
