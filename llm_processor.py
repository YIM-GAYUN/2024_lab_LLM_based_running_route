# LLM을 통해 사용자 입력을 분석하여 json으로 반환

import json
from openai import OpenAI
client = OpenAI()

def extract_route_features(user_input):
    # 사용자의 요구사항을 저장할 딕셔너리
    running_requirements = {
        'Distance': None,
        'Inclination': None,
        'Environment': None,
        'Additional preference': None
    }
    
    # LLM에 보낼 프롬프트
    prompt = f"""The user has provided the following running-related requirements: "{user_input}".
    Please extract the following details:
    - Distance
    - Inclination (flat or inclined)
    - Environment (park, river, etc.)
    - Additional preference (if mentioned, such as nearby landmarks or surroundings or facilities)
    Provide the details in JSON format.
    ex) 'Distance': '2km', 'Inclination': 'flat', 'Enviornment': 'park', 'Additional preference': None
    """

    try:
        response = client.chat.completions.create( # 특정한 대화 스타일로 LLM과 상호작용하는 메서드
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant designed to help users by extracting specific features from their running-related requirements. Your task is to identify key attributes such as distance, inclination, environment, and any additional preferences mentioned by the user. Respond only with these details in a structured JSON format without any extra information."},
                {"role": "user", "content": prompt}
            ],
        )

        # 응답 내용 추출
        result = response.choices[0].message.content.strip() # response는 모델의 응답, 현재 응답 하나를 뽑기 때문에 choices[0], strip은 문자열에서 공백 제거
        
        # JSON 파싱 시도
        parsed_result = json.loads(result)

        # running_requirements 업데이트
        running_requirements.update(parsed_result)

        # JSON 형태로 반환
        return running_requirements
    except Exception as e:
        return {"error": str(e)}