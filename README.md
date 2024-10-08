# LLM_ChatBot

![llama3-8b](https://github.com/user-attachments/assets/d29699f8-e9d2-43b9-912a-cdf9c40c22f6)

**Complete instruction on how to run python script on running Meta-Llama-3-8B-Instruct with Intel Xeon 4th Gen and Newer with IPEX**

**Steps to run this demo:**
1) Clone this project into your local directory:
```   
   git clone https://github.com/allenwsh82/LLM_ChatBot.git
```
   
2) Create a new virtual environment inside the project which you just clone:
```
   python -m venv llm_chatbot_env
```

3) Activate the virtual environment which you just created:
```
   source llm_chatbot_env/bin/activate
```

4) Install the dependencies by running:
  ```
  pip install -r requirements.txt
```

5) If you go to the inference_llama3_8b_bf16_ipex.py script, you will notice where two lines of code are added to enable AMX AI Accelerator to boost up performance:

###########################################################################################################
#Use IPEX

import intel_extension_for_pytorch as ipex

model = ipex.optimize(model.eval(), dtype=torch.bfloat16, inplace=True, level="O1", auto_kernel_selection=True)

###########################################################################################################
   
6) Now you have setup everything and you can run the script:
```   
   python inference_llama3_8b_bf16_ipex.py
```

Inference example snapshot:

![llama3_8b_inference](https://github.com/user-attachments/assets/23c7d358-3fee-4380-9599-c0faef6faafc)

7) If you want to further optimize the inference performance, just quatize the model to INT4 easily with Intel ipex-llm. 
```
   python inference_llama3_8b_INT4_IPEX_LLM.py
```
Inference example snapshot:

![ipex-llm_llama3-8b](https://github.com/user-attachments/assets/6e12b530-da5e-480d-b0f2-07b41f413512)





   
