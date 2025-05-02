#Using pre-train transformer 
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

def paraphrase_text(text: str):
    model_name = "t5-large"

    try:
        # Load tokenizer and model 
        tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        model = T5ForConditionalGeneration.from_pretrained(model_name)

        # Prepare input
        input_text = f"paraphrase: {text}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        # Generate paraphrase
        with torch.no_grad():
            output = model.generate(input_ids, max_length=100, num_beams=5, do_sample=True, top_p=0.95, early_stopping=True)

        # Decode output
        paraphrased_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return paraphrased_text

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    sample=input('Enter the text to paraphrase')
    #sample = "Machine learning is revolutionizing many industries by automating decision-making."
    print(paraphrase_text(sample))
