#!/usr/bin/env python
# coding: utf-8

# In[41]:


from transformers import AutoTokenizer, T5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("t5-large")
model = T5ForConditionalGeneration.from_pretrained("gsdas/qct5")


def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    t = tokenizer.batch_decode(res, skip_special_tokens=True)[0].replace("<pad>","").replace("</s>","").strip()
    if ":" not in t:
        return "ENTY:other"
    return t



test_eg=["ENTY:plant\tWhat is Australia 's national flower ?",
"DESC:reason\tWhy does the moon turn orange ?",
"DESC:def\tWhat is autism ?",
"LOC:city\tWhat city had a world fair in 1900 ?",
"HUM:ind\tWhat person 's head is on a dime ?",
"NUM:weight\tWhat is the average weight of a Yellow Labrador ?"
]

for eg in test_eg:

    tc = eg.split("\t")[0]
    q = eg.split("\t")[1].strip().lower()

    op = run_model(q)
    print ("\n"+q)
    print ("T5 response: "+op)
    print ("True Class: "+tc)
