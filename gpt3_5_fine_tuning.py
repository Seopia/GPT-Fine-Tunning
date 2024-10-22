
from openai import OpenAI
from pathlib import Path
import tiktoken
import json


today_exchange_rate = 1380.40

api_key = 'API 키'
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
client = OpenAI(api_key=api_key)
file_path = "train.jsonl"  # 훈련 데이터

def use_default_gpt_model(user_prompt):
    if prompt=="e":
        exit(0)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )
    print(chat_completion.choices[0].message.content)

def use_fine_tuning_model(model, user_prompt):
    res = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": user_prompt,
        }]
    )
    print(res.choices[0].message.content.strip())

def fine_tune():
    if count_token():
        train_data_file = client.files.create(
            file=Path(file_path),
            purpose="fine-tune",
        )
        # print(f"File ID: {train_data_file.id}")
        # print(f"File : {train_data_file}")
        # 파인 튜닝 작업을 생성하고 결과를 저장합니다.
        fine_tune_job = client.fine_tuning.jobs.create(
            model="gpt-3.5-turbo",
            training_file=train_data_file.id,
        )

        # 생성된 파인 튜닝 작업의 ID를 사용하여 상태를 검색합니다.
        job_id = fine_tune_job.id  # 작업 ID를 얻습니다.
        print(f"Fine-tune job ID: {job_id}")  # 작업 ID를 출력합니다.

        # 작업 ID를 사용하여 파인 튜닝 작업의 상태를 검색합니다.
        job_status = client.fine_tuning.jobs.retrieve(job_id)
        print(job_status)  # 작업 상태를 출력합니다.

    else:
        print("작업을 취소했습니다.")
        return


def count_token():
    total_tokens = 0

    # 토큰 계산 함수
    def count_tokens(text):
        tokens = encoding.encode(text)
        return len(tokens)

    # 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)  # JSONL에서 각 줄을 JSON으로 읽음
            for message in data['messages']:
                role = message['role']
                content = message['content']

                if role == "user":
                    prompt_tokens = count_tokens(content)
                    total_tokens += prompt_tokens
                elif role == "assistant":
                    completion_tokens = count_tokens(content)
                    total_tokens += completion_tokens

    # 총 토큰 수 출력
    print(f"총 토큰 수 : {total_tokens}")

    # 비용 계산
    usd = total_tokens * (0.0015 / 1000)
    print("====== 가격 =====")
    print(f"{usd:.6f} USD")
    print(f"{usd * today_exchange_rate:.2f} 원")  # 환율을 적용한 원화 값


    user_input = input("계속 하시겠습니까? (Y, N) : ")
    if user_input.upper() == 'Y':
        return True
    elif user_input.upper() == 'N':
        return False

select_mode = input("(1) 기본 GPT3.5\n(2)파인 튜닝\n : ")
if select_mode == '1':
    while True:
        prompt = input("Prompt : ")
        use_default_gpt_model(prompt)
elif select_mode == '2':
    while True:
        user_input = input("(1) 파인튜닝 시키기\n(2) 파인튜닝 AI와 대화하기\n(3) 파인튜닝 모델 이름 살펴보기\n(4) 종료\n : ")
        if user_input == '1':
            fine_tune()
            print("파인 튜닝은 시간이 오래 걸립니다. https://platform.openai.com/finetune/ 에서 확인해보세요.")
        elif user_input == '2':
            user_prompt = input("Prompt : ")
            model_name = input("Model Name : ")
            use_fine_tuning_model(model_name, user_prompt)
        elif user_input == '3':
            for model in client.fine_tuning.jobs.list():
                if model.error.code is None:
                    print("대화 가능한 모델은 : ",model.fine_tuned_model, "입니다.")
        elif user_input == 'exit':
            exit(0)