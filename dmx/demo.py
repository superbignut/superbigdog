from zhipuai import ZhipuAI
# import pyttsx3
import sys

sys.path.append("C:\conda\envs\dmx\Lib\site-packages\zhipuai")

conversation_id = None

while True:
    #output=input("user:")
    output="站在狗的角度，分析说句话对狗的态度属于下面的哪类，赞美，批评还是辱骂，只用这三个词中的一个回答。"
    output += input("input:")
    api_key = "299adac92d9b98c139f22fa1e22a8b2c.t7LzNyfNX49gsShG"
    url = "https://open.bigmodel.cn/api/paas/v4"
    client = ZhipuAI(api_key=api_key, base_url=url)
    prompt = output
    generate = client.assistant.conversation(
        assistant_id="659e54b1b8006379b4b2abd6",
        conversation_id=conversation_id,
        model="glm-4-assistant",
        messages=[
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": prompt
                }]
            }
        ],
        stream=True,
        attachments=None,
        metadata=None
    )
    output = ""
    for resp in generate:
        if resp.choices[0].delta.type == 'content':
            output += resp.choices[0].delta.content
            conversation_id = resp.conversation_id
    print(output)
