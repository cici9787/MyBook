from modelscope import AutoModelForCausalLM, AutoTokenizer
device = "cuda"
print("device:", device)

path = "/root/autodl-tmp/Qwen2-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype = "auto",
        device_map = "auto"
        )

print("model:", model)
tokenizer = AutoTokenizer.from_pretrained(path)
prompt = "五一应该去哪玩？"

message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
        ]

text = tokenizer.apply_chat_template(
        message,
        tokenize = False,
        add_generation_prompt = True
        )

print("text:", text)

model_inputs = tokenizer([text], return_tensors="pt").to(device)
print("model_inputs:", model_inputs)

generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens = 512
        )

print("generated_ids:", generated_ids)

generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids,
            generated_ids)
        ]

print("generated_ids:", generated_ids)
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("response:", response)

# cd /root/.ssh
# cat id_rsa.pub