import os
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, Trainer, TrainingArguments

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# KoGPT 모델 로드
model = GPT2LMHeadModel.from_pretrained('taeminlee/kogpt2')

# PreTrainedTokenizerFast를 사용하여 토크나이저 로드
# 토크나이저란?
# 텍스트 데이터를 수치 데이터로 변환하는 도구이다.
# 1. 문장을 단어, 서브워드 단위로 쪼개서 토큰으로 만드는 토크나이징 과정
# 2. 토큰을 모델이 이해할 수 있는 정수 ID로 변환
# 3. 정수 ID를 다시 텍스트로 변환할 수도 있다.
tokenizer = PreTrainedTokenizerFast.from_pretrained('taeminlee/kogpt2')

# 학습용 데이터 로드 및 토큰화
def load_text_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    dataset = []
    for line in lines:
        if line.strip():
            # A: 를 기준으로 나눠서 변수에 할당
            question, answer = line.strip().split('A:')
            dataset.append({"text": question + ' A:' + answer})
    return dataset

# 한글 데이터셋 로드
dataset = load_text_dataset('dataset_ko.txt')

#
def tokenize_function(examples):
    # 딕셔너리에 text를 토큰화 하며 최대 길이는 128이고, 128보다 작으면 채운다.
    tokens = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# 토큰화 한 것들을 리스트로 저장
tokenized_datasets = [tokenize_function(example) for example in dataset]

# 학습용 PyTorch 데이터셋 생성
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.data[idx]["input_ids"])
        attention_mask = torch.tensor(self.data[idx]["attention_mask"])
        labels = torch.tensor(self.data[idx]["labels"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

train_dataset = TextDataset(tokenized_datasets)

# 학습 설정
training_args = TrainingArguments(
    output_dir=".venv/results",
    eval_strategy="no",
    per_device_train_batch_size=4,
    num_train_epochs=30,
    learning_rate=5e-5,
    save_steps=1000,
    save_total_limit=2,
    fp16=True,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# 모델 학습
trainer.train()

# 학습된 모델 저장
model.save_pretrained("./kogpt2-finetuned")
tokenizer.save_pretrained("./kogpt2-finetuned")


# 학습된 모델과 토크나이저 로드
model = GPT2LMHeadModel.from_pretrained("./kogpt2-finetuned")
tokenizer = PreTrainedTokenizerFast.from_pretrained("./kogpt2-finetuned")

# 모델을 평가 모드로 설정
model.eval()
model.to('cuda')  # GPU 사용

while True:
    prompt = input("Q: ")
    if prompt == '':
        print("종료")
        break

    # 텍스트를 토큰화하고, 텐서로 반환한 후 GPU로 전송한다.
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

    # 답변 생성하기
    outputs = model.generate(
        inputs['input_ids'],    # 숫자로 변환한 토큰 ID는 학습된 사전에서 텍스트를 찾을 수 있게 해준다.
        attention_mask=inputs['attention_mask'],    # 실제로 유효한 토큰인지 구분하는 것. 쓸모 없는 것을 없애준다.
        max_length=50,  # 최대 50 토큰만 생성할 수 있다.
        temperature=0.8,    # 단어를 선택할 때 확률 분포를 조정한다. 낮으면 신중하고, 높으면 더 무작위로 선택한다. 텍스트 생성의 다양성이다.
        top_k=50,           # 단어 생성 과정에서 후보로 고려할 단어의 수이다.
        top_p=0.9,          # 생성한 단어들의 확률이 0.9가 될 때 까지 상위 단어들을 선택하게 한다.
        do_sample=True,     # 항상 같은 답변을 생성하지 않게 해준다.
        pad_token_id=tokenizer.eos_token_id
    )

    # 생성된 응답을 텍스트로 디코드
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    first_sentence = response.split('.')[0] + '.'

    res = first_sentence.split(': ')
    if len(res)>=2:
        print(res[1])
    else:
        print('이상한 답변? : ',first_sentence)