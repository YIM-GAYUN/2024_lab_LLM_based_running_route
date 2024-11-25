# Flask 서버를 실행하고, 사용자로부터 요청을 받아서 LLM 처리와 경로 생성을 연결하는 메인 파일

from flask import Flask, request, jsonify
from llm_processor import extract_route_features
from osm_processor import get_osm_features
from visualize import visualize_route

app = Flask(__name__) # Flask 인스턴스를 생성하여 웹 서버 시작

@app.route('/generate-route', methods=['POST']) # POST 요청을 받을 경로
def get_route(): # POST 요청을 받을 때마다 자동으로 함수 호출
    # JSON 데이터 받기
    data = request.json

    # 데이터가 제공되지 않았을 때 오류 응답
    if not data:
        return jsonify({"error": "No data provided"}), 400 # 400 에러: 요청이 잘못 되었음

    # 사용자 입력 및 출발지, 도착지 확인
    user_input = data.get('input', '')
    start = data.get('start', {})
    end = data.get('end', {})

    # 출발지 또는 도착지가 제공되지 않았을 때 오류 응답
    if not start and not end:
        return jsonify({"error": "Start or End locations must be provided"}), 400

    try:
        # LLM을 사용해 특징 추출
        features = extract_route_features(user_input)
        app.logger.info("extract_route_features")
        app.logger.debug(features)
    except Exception as e:
        return jsonify({"error": f"Feature extraction failed: {str(e)}"}), 500 # 500 에러: 서버 오류
    
    try:
        get_osm_features(start, end, features)
    except Exception as e:
        return jsonify({"error": f"Get osm failed: {str(e)}"}), 500

if __name__ == '__main__': # app.py 파일이 직접 실행될 때
    app.run(debug=True) # 코드를 수정할 때마다 서버가 자동으로 재시작