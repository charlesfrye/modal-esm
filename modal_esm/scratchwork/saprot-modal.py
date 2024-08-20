import modal

app = modal.App("saprot-esm")

def download_model():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache
    adapter_input = "SaProtHub/Model-EYFP-650M"
    base_model = "westlake-repl/SaProt_650M_AF2"
    
    snapshot_download(repo_id=adapter_input, repo_type="model")
    snapshot_download(repo_id=base_model, repo_type="model")
    move_cache()


image = (
    modal.Image.debian_slim()
    .pip_install("uv")
    .run_commands("uv pip install  --system --compile-bytecode torch transformers==4.28 huggingface_hub pandas tqdm peft")
    .run_function(download_model)
)

@app.function(image=image, gpu="any")
def predict_protein_sequence(sequences):
    import torch
    from transformers import EsmTokenizer
    from peft import PeftModelForSequenceClassification
    from transformers import EsmForSequenceClassification
    
    # Set up model configuration
    # adapter_path = "/root/.cache/huggingface/hub/models--SaProtHub--Model-EYFP-650M"
    adapter_path = "SaProtHub/Model-EYFP-650M"
    base_model_name = "westlake-repl/SaProt_650M_AF2"

    base_model = EsmForSequenceClassification.from_pretrained(base_model_name, num_labels=1,)
    model = PeftModelForSequenceClassification.from_pretrained(
        base_model,
        adapter_path,
        )

    tokenizer = EsmTokenizer.from_pretrained(base_model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Process sequences
    outputs_list = []
    for seq in sequences:
        sa_seq = '#'.join(seq) + '#'
        inputs = tokenizer(sa_seq, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        outputs_list.append(outputs)

    return outputs_list

@app.local_entrypoint()
def main():
    sequences = [
        "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
        "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFGYGLQCFARYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSYQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK",
    ]
    results = predict_protein_sequence.remote(sequences)
    print(results)