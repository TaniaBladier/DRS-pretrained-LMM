# a case of drs-text generation
from tokenization_mlm import MLMTokenizer
from transformers import MBartForConditionalGeneration
import torch
"""
# For DRS parsing, src_lang should be set to en_XX, de_DE, it_IT, or nl_XX
tokenizer = MLMTokenizer.from_pretrained('laihuiyuan/DRS-LMM', src_lang='<drs>')
model = MBartForConditionalGeneration.from_pretrained('laihuiyuan/DRS-LMM')

# gold text: The court is adjourned until 3:00 p.m. on March 1st.
inp_ids = tokenizer.encode(
    "court.n.01 time.n.08 EQU now adjourn.v.01 Theme -2 Time -1 Finish +1 time.n.08 ClockTime 15:00 MonthOfYear 3 DayOfMonth 1",
    return_tensors="pt")
print("13", inp_ids)


# For DRS parsing, the forced bos token here should be <drs> 
foced_ids = tokenizer.encode("en_XX", add_special_tokens=False, return_tensors="pt")
print("15", foced_ids)
outs = model.generate(input_ids=inp_ids, forced_bos_token_id=foced_ids.item(), num_beams=5, max_length=150)
print("17", outs)
text = tokenizer.decode(outs[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("19", text)

"""

sent_to_eval =  "She's donating money for cancer research."

# For DRS parsing, src_lang should be set to en_XX, de_DE, it_IT, or nl_XX
tokenizer2 = MLMTokenizer.from_pretrained('laihuiyuan/DRS-LMM', src_lang='en_XX')
model2 = MBartForConditionalGeneration.from_pretrained('laihuiyuan/DRS-LMM')

# gold text: The court is adjourned until 3:00 p.m. on March 1st.
inp_ids = tokenizer2.encode(
    sent_to_eval,
    return_tensors="pt")
print("321", inp_ids)

foced_ids = tokenizer2.encode("<drs>", add_special_tokens=False, return_tensors="pt")
print("151", foced_ids)
outs = model2.generate(input_ids=inp_ids, forced_bos_token_id=foced_ids.item(), num_beams=5, max_length=150)
print("171", outs)
text = tokenizer2.decode(outs[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("191", text)

#gold DRS: court.n.01 time.n.08 EQU now adjourn.v.01 Theme -2 Time -1 Finish +1 
#           time.n.08 ClockTime 15:00 MonthOfYear 3 DayOfMonth 1


# For DRS parsing, src_lang should be set to en_XX, de_DE, it_IT, or nl_XX
tokenizer3 = MLMTokenizer.from_pretrained('/home/tb/Programs/DRS-pretrained-LMM/checkpoints/mbart-large-50', src_lang='en_XX')
model3 = MBartForConditionalGeneration.from_pretrained("/home/tb/Programs/DRS-pretrained-LMM/checkpoints/mbart-large-50")
model_dir = 'checkpoints/mlm_sft.chkpt'
model3.load_state_dict(torch.load(model_dir))
# gold text: The court is adjourned until 3:00 p.m. on March 1st.
inp_ids = tokenizer3.encode(
    sent_to_eval,
    return_tensors="pt")
print("32", inp_ids)

foced_ids = tokenizer3.encode("<drs>", add_special_tokens=False, return_tensors="pt")
print("15", foced_ids)
outs = model3.generate(input_ids=inp_ids, forced_bos_token_id=foced_ids.item(), num_beams=5, max_length=150)
print("17", outs)
text = tokenizer3.decode(outs[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("19", text)