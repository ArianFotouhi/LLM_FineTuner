from transformers import AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer

repo_id = "facebook/opt-350m"
ds = "imdb"


dataset = load_dataset(ds, split="train")
model = AutoModelForCausalLM.from_pretrained(repo_id)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
)

trainer.train()

